"""Step 2 (RAG) + Step 3 (conversation, verification, tools, handover).

ZentroAgent.answer_once(question) -> Step-2-style single-turn RAG answer,
    returns (reply_text, used_section_ids).
ZentroAgent.respond(customer_message) -> Step-3 conversational turn that
    remembers state (order id, verification) across calls on the same instance.
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SYSTEM_PROMPT_BASE, call_llm, call_llm_chat, TODAY
from src.kb import KBIndex, load_products
from src import datemath
from tools.tools import lookup_order, create_ticket

PAYMENT_METHOD_KEYWORDS = ["credit card", "debit card", "card", "upi", "net banking",
                           "netbanking", "wallet", "bank transfer", "neft"]

RAG_SYSTEM_SUFFIX = """

You will be given retrieved document excerpts, each tagged with its [SECTION-ID], \
doc_type and status. Use ONLY these excerpts to answer. If a current policy and a \
superseded one disagree, follow the current one unless the superseded one still \
explicitly applies (e.g. an order placed/delivered before a policy's effective_from \
date). If the excerpts do not answer the question, say you don't know and offer a \
human colleague -- do not guess. This applies even when a nearby/related excerpt \
exists: do not infer or state a specific rule, number or availability (e.g. a \
delivery slot, a fee, a deadline) unless an excerpt literally says it -- a related \
section being retrieved does not mean it answers this specific question. \
When an excerpt gives a specific number (a count, a day limit, a fee), include that \
exact number in your answer rather than only describing the process in general terms.

If the prompt includes a CATALOG_FACTS or COMPUTED_DATE_FACTS line, that information \
is already correct and already looked up for you -- use it directly, exactly as given. \
Never say you lack catalog or date data when one of these lines is present, and never \
recompute a business-day date yourself when COMPUTED_DATE_FACTS already gives it.

After your reply, on a new line, output the section IDs you actually used, like:
SOURCES: RET-2, REF-1
"""

AGENT_SYSTEM = SYSTEM_PROMPT_BASE + f"""

Today's date is {TODAY.strftime('%d %B %Y')}. Use it for every delay/deadline calculation.

You have two tools available, described to you as JSON results when you ask for them:
- lookup_order(order_id, verification_value=None): call with verification_value=None \
first. Only after the customer gives their order's last-4 phone digits OR 6-digit \
delivery pincode, call again with that value. Never reveal order status, items, \
address or amounts before a verified lookup succeeds. Allow at most one retry on a \
failed verification; after a second failure, do not share order details.
- create_ticket(summary, order_id=None, reason=None, intent=None): use for human \
handover (see agent_escalation_handoff.md). Only say a ticket was created if the \
tool result says created=True.

Never invent order data, and never claim to have looked something up or created a \
ticket unless the tool result (shown to you as TOOL_RESULT) actually confirms it.

When a message includes COMPUTED_DATE_FACTS, those business-day and return-window \
numbers are already correct -- read them, do not recompute dates or day counts yourself.

Hand over to a human colleague (see agent_escalation_handoff.md [ESC-1]) whenever any \
of these is true: a damaged/defective/wrong item worth more than Rs 10,000; the customer \
reports an order, payment or account activity they did not make; the customer mentions \
legal action, a consumer court or police; this is the 3rd+ contact about the same issue \
within 7 days (contacts_last_7_days >= 2 in the order data); a refund is more than 3 \
business days past its maximum timeline; the customer explicitly asks for a human; or a \
product safety risk (swelling battery, burning smell, sparks, injury). Never approve, \
promise or perform the thing the customer is asking for in these cases -- only a human can.

When you decide to hand over, after your customer-facing reply (and after any SOURCES \
line), add one more new line starting with "ESCALATE:" followed by a one-paragraph ticket \
summary covering: the customer's issue, the detected intent, the order ID if any, what \
the customer has told you, what you already checked, and the reason for escalation (see \
agent_escalation_handoff.md [ESC-3]). Do not add this line when no handover is needed. \
Your customer-facing reply itself should just say a human colleague will follow up -- \
never promise an outcome, amount or timeframe the policy doesn't state.
"""


def _computed_date_facts(order: dict) -> str:
    """Pre-compute business-day/return-window facts so the LLM never has to do
    date arithmetic itself -- it just reads these off."""
    facts = []
    try:
        if order.get("promised_date"):
            if TODAY > datemath._parse(order["promised_date"]) and not order.get("delivered_on"):
                late_days = datemath.calendar_days_between(order["promised_date"], TODAY)
                facts.append(f"Order is {late_days} calendar day(s) past its promised_date and not yet delivered.")
        if order.get("delivery_fee_inr") == 79 and order.get("delivered_on") and order.get("promised_date"):
            if datemath._parse(order["delivered_on"]) > datemath._parse(order["promised_date"]):
                facts.append("This was an Express order delivered after its promised_date -> DEL-4 wallet credit applies (unless the customer is a Plus member, who paid no express fee).")
        if order.get("return_sku") and order.get("delivered_on"):
            items = order.get("items", [])
            cat = next((i.get("return_category") for i in items if i.get("sku") == order["return_sku"]), None)
            if cat:
                window = datemath.return_window_days(cat, order["delivered_on"])
                within = datemath.is_within_return_window(cat, order["delivered_on"])
                facts.append(f"Return item category={cat}: window is {window} calendar day(s) from delivery "
                              f"({order['delivered_on']}); as of today the order is "
                              f"{'WITHIN' if within else 'OUTSIDE'} that window.")
        if order.get("return_requested_on") and not order.get("return_refund_initiated_on"):
            bdays = datemath.business_days_between(order["return_requested_on"], TODAY)
            facts.append(f"{bdays} business day(s) have passed since the return was requested; refund is initiated within 2 business days of PICKUP (not request), see REF-1.")

        # generic refund-arrival estimate by this order's actual payment method,
        # in case the customer asks "how long will my refund take" for this order
        pm = order.get("payment_method")
        if pm:
            pm_key = {"card": "credit card", "upi": "upi", "netbanking": "net banking",
                      "wallet": "wallet", "cod": "cod_bank"}.get(pm, pm)
            window = datemath.REFUND_RANGES_BUSINESS_DAYS.get(pm_key)
            if window:
                lo, hi = window
                if lo == 0:
                    facts.append(f"This order's payment method is {pm}: refunds to it are instant (within 2 hours).")
                else:
                    facts.append(f"This order's payment method is {pm}: refunds normally take {lo} to {hi} business days after initiation.")
        if order.get("is_plus"):
            facts.append("Customer is a ZentroMart Plus member: refund for a return is initiated AT PICKUP, not after QC.")
    except Exception:
        pass  # never let date-math errors break the conversation
    return " ".join(facts)


def _return_eligibility_facts(question: str, order: dict) -> str:
    """Can-I-return-this-item questions asked BEFORE a return is filed (so
    order['return_sku'] is empty) still need a real deadline -- match any
    product named in the question against the catalog, then compute the
    window/deadline/fee from the order's own delivered_on and placed_at."""
    if not order.get("delivered_on"):
        return ""
    facts = []
    try:
        for p in load_products():
            title = p["title"]
            base = title.split(",")[0].strip()
            key = " ".join(base.split()[:4]).lower()
            if not (title.lower() in question.lower() or base.lower() in question.lower()
                    or (len(key) > 8 and key in question.lower()) or p["sku"] in question):
                continue
            cat = p["return_category"]
            deadline = datemath.return_deadline(cat, order["delivered_on"], order.get("placed_at"))
            if deadline is None:
                facts.append(f"{title} (category={cat}) is not returnable for change of mind (RET-5).")
                continue
            within = datemath.TODAY <= deadline
            fee = datemath.reverse_pickup_fee_inr(cat, bool(order.get("is_plus")))
            facts.append(
                f"{title} (category={cat}): delivered {order['delivered_on']}, return deadline is "
                f"{deadline.strftime('%d %B %Y')} ({'still WITHIN' if within else 'OUTSIDE'} the window as of today). "
                f"Reverse pickup fee for this return: Rs {fee}."
            )
    except Exception:
        pass
    return " ".join(facts)


def _catalog_facts(question: str) -> str:
    """Surface actual catalog data (warranty, assured, price, category) for any
    product named or SKU'd in the question -- retrieval alone only biases which
    kb sections come back, it never puts the product's own row in front of the
    LLM, so SKU-specific questions were otherwise unanswerable."""
    facts = []
    q_lower = question.lower()
    for p in load_products():
        title = p["title"]
        base = title.split(",")[0].strip()
        key = " ".join(base.split()[:4]).lower()
        if (title.lower() in q_lower or base.lower() in q_lower
                or (len(key) > 8 and key in q_lower) or p["sku"] in question):
            facts.append(
                f"Catalog row for {p['sku']} ({title}): return_category={p['return_category']}, "
                f"price_inr={p['price_inr']}, fulfilled_by={p['fulfilled_by']}, "
                f"assured={p['assured']}, warranty_months={p['warranty_months']}."
            )
    return " ".join(facts)


def _refund_timing_facts(question: str) -> str:
    """If the question names a payment method and a date (e.g. 'my credit card
    refund was initiated on 17 October 2026'), compute the real earliest/latest
    arrival date with business-day math instead of leaving that arithmetic to
    the LLM (which reliably miscounts business days around holidays)."""
    d = datemath.find_date_in_text(question)
    if not d:
        return ""
    q_lower = question.lower()
    method = next((m for m in PAYMENT_METHOD_KEYWORDS if m in q_lower), None)
    if not method:
        return ""
    return datemath.refund_window_for(method, d)


def _format_chunks(chunks):
    return "\n\n".join(
        f"[{c.section_id}] {c.heading} (doc_type={c.doc_type}, status={c.status}, "
        f"effective_from={c.effective_from})\n{c.text}"
        for c in chunks
    )


class ZentroAgent:
    def __init__(self):
        self.kb = KBIndex()
        self.history = []          # [{"role": .., "content": ..}]
        self.order_id = None
        self.verified = False
        self.verify_attempts = 0
        self.last_tool_calls = []   # tool calls made during the most recent respond() turn
        self.last_ticket_result = None

    # ---------- Step 2: single-turn RAG ----------
    def answer_once(self, question: str, k: int = 6):
        chunks = self.kb.retrieve(question, k=k)
        extras = []
        catalog = _catalog_facts(question)
        if catalog:
            extras.append(f"CATALOG_FACTS: {catalog}")
        refund = _refund_timing_facts(question)
        if refund:
            extras.append(f"COMPUTED_DATE_FACTS: {refund}")
        extra_block = ("\n".join(extras) + "\n\n") if extras else ""
        prompt = (f"{extra_block}Retrieved excerpts:\n\n{_format_chunks(chunks)}\n\n"
                  f"Customer question: {question}")
        raw = call_llm(SYSTEM_PROMPT_BASE + RAG_SYSTEM_SUFFIX, prompt)
        reply, sources = self._split_sources(raw)
        return reply, sources or [c.section_id for c in chunks]

    @staticmethod
    def _split_sources(raw: str):
        m = re.search(r"SOURCES:\s*(.*)$", raw, re.M)  # .* not .+ -- also strip an empty tag
        if not m:
            return raw.strip(), []
        reply = raw[:m.start()].strip()
        ids = [s.strip() for s in m.group(1).split(",") if s.strip()]
        return reply, ids

    # ---------- Step 3: multi-turn conversation with tools ----------
    def respond(self, customer_message: str) -> str:
        """One conversational turn. Keeps order id / verification in self.* only
        (never echoed back to the customer, per agent_identity_verification.md).
        Tool calls made this turn are recorded in self.last_tool_calls for the
        evaluator / caller to inspect."""
        self.last_tool_calls = []
        self.last_ticket_result = None
        # order IDs are read/written space-grouped in fours, e.g. "8978 2837"
        # (see agent_conversation_style.md [STY-3]), as well as sometimes plain
        oid_match = re.search(r"\b(\d{4})\s?(\d{4})\b", customer_message)
        if oid_match and not self.order_id:
            self.order_id = oid_match.group(1) + oid_match.group(2)

        tool_context = ""
        if self.order_id and not self.verified:
            proof_match = re.search(r"\b(\d{4}|\d{6})\b", customer_message)
            if proof_match and oid_match is None:
                # a 4- or 6-digit number that isn't the order id itself -> treat as proof
                proof = proof_match.group(0)
                result = lookup_order(self.order_id, verification_value=proof)
                self.last_tool_calls.append(f"lookup_order(order_id={self.order_id}, verification_value={proof})")
                self.verify_attempts += 1
                if result.get("verified"):
                    self.verified = True
                    self._order_data = result
                else:
                    if self.verify_attempts >= 2:
                        tool_context = ("TOOL_RESULT lookup_order: verification failed twice. "
                                         "Do not share order details; offer general help or a human.")
                    else:
                        tool_context = "TOOL_RESULT lookup_order: verification failed, one retry allowed."
            else:
                pending = lookup_order(self.order_id)
                self.last_tool_calls.append(f"lookup_order(order_id={self.order_id})")
                tool_context = f"TOOL_RESULT lookup_order: {json.dumps(pending)}"
        elif self.order_id and self.verified:
            computed = _computed_date_facts(self._order_data)
            eligibility = _return_eligibility_facts(customer_message, self._order_data)
            if eligibility:
                computed = (computed + " " + eligibility).strip()
            tool_context = (f"TOOL_RESULT lookup_order (verified, cached): "
                             f"{json.dumps(self._order_data, default=str)}\n"
                             f"COMPUTED_DATE_FACTS (already correct, do not recompute): {computed}")

        chunks = self.kb.retrieve(customer_message, k=5)
        context = _format_chunks(chunks)
        user_turn = (f"{tool_context}\n\nRetrieved excerpts:\n{context}\n\n"
                      f"Customer: {customer_message}")

        self.history.append({"role": "user", "content": user_turn})
        raw = call_llm_chat(AGENT_SYSTEM, self.history, max_tokens=500)
        self.history.append({"role": "assistant", "content": raw})  # keep full raw turn in context

        reply, escalation_summary = self._split_escalation(raw)

        if escalation_summary:
            result = self.escalate(escalation_summary, intent=None)
            self.last_tool_calls.append("create_ticket()")
            self.last_ticket_result = result

        return reply

    @staticmethod
    def _split_escalation(text: str):
        m = re.search(r"ESCALATE:\s*(.+)$", text, re.S)
        if not m:
            return text.strip(), None
        reply = text[:m.start()].strip()
        summary = m.group(1).strip()
        return reply, summary

    def escalate(self, summary: str, reason: str = None, intent: str = None):
        # create_ticket requires a real, >=20-char summary (see tools/tools.py);
        # ESC-3 requires it cover the issue, intent, order id, what was checked
        # and the escalation reason -- the LLM is prompted to write exactly that.
        return create_ticket(summary, order_id=self.order_id, reason=reason, intent=intent)


if __name__ == "__main__":
    a = ZentroAgent()
    reply, sources = a.answer_once("How long do I have to return a pair of shoes?")
    print(reply)
    print("sources:", sources)
