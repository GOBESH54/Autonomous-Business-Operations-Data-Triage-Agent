# Escalation Procedures

## Tier Structure

### Tier 1 — Front-line Support
- Handle password resets, billing questions, basic configuration.
- **Resolution target**: 80% of tickets within SLA.
- **Escalate when**: Issue requires code-level investigation or infrastructure access.

### Tier 2 — Technical Support Engineers
- Handle API issues, integration bugs, performance investigations.
- **Resolution target**: 90% within SLA.
- **Escalate when**: Root cause is in core platform or affects multiple customers.

### Tier 3 — Engineering On-Call
- Handle production incidents, data integrity issues, security events.
- **Must acknowledge within**: 15 minutes for critical, 1 hour for high.

## Escalation Triggers

- Customer is an **enterprise** account → auto-priority boost by one level.
- Ticket mentions **data loss**, **security**, or **outage** → auto-escalate to Tier 3.
- Customer has **3+ open tickets** → flag for account health review.
- SLA at **75% consumed** with no resolution path → escalate to next tier.

## Communication Protocol

- Update customer every **2 hours** for critical tickets.
- Update customer every **4 hours** for high-priority tickets.
- Internal handoff notes must include: reproduction steps, logs collected, actions taken.
