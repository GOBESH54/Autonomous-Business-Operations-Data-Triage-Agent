"""
Seed script: Initializes data/business.db with realistic e-commerce/SaaS data
and creates sample SOP/policy markdown documents in data/policies/.
"""
import sqlite3
import os
import random
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "business.db")
POLICIES_DIR = os.path.join(BASE_DIR, "data", "policies")


def seed_database():
    """Create and populate the SQLite database with sample business data."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Remove existing DB for fresh seed
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── Create tables ──────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE customers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    NOT NULL UNIQUE,
            plan        TEXT    NOT NULL DEFAULT 'free',
            signup_date TEXT    NOT NULL,
            region      TEXT    NOT NULL
        );

        CREATE TABLE orders (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id   INTEGER NOT NULL REFERENCES customers(id),
            product       TEXT    NOT NULL,
            amount        REAL    NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'pending',
            order_date    TEXT    NOT NULL,
            delivery_date TEXT
        );

        CREATE TABLE support_tickets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     INTEGER NOT NULL REFERENCES customers(id),
            subject         TEXT    NOT NULL,
            description     TEXT    NOT NULL,
            priority        TEXT    NOT NULL DEFAULT 'medium',
            status          TEXT    NOT NULL DEFAULT 'open',
            sla_deadline    TEXT    NOT NULL,
            created_at      TEXT    NOT NULL,
            resolved_at     TEXT
        );

        CREATE TABLE action_tickets (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id         INTEGER NOT NULL REFERENCES customers(id),
            issue_summary       TEXT    NOT NULL,
            priority            TEXT    NOT NULL DEFAULT 'medium',
            resolution_action   TEXT    NOT NULL,
            status              TEXT    NOT NULL DEFAULT 'created',
            created_at          TEXT    NOT NULL
        );
    """)

    # ── Seed customers ─────────────────────────────────────────────
    customers = [
        ("Alice Johnson", "alice@acmecorp.com", "enterprise", "2024-01-15", "US-East"),
        ("Bob Smith", "bob@techstartup.io", "pro", "2024-03-22", "US-West"),
        ("Carlos Rivera", "carlos@globalretail.com", "enterprise", "2023-11-08", "EU-West"),
        ("Diana Chen", "diana@innovatehq.com", "free", "2024-06-10", "APAC"),
        ("Eve Martinez", "eve@designco.com", "pro", "2024-02-28", "US-East"),
        ("Frank Okafor", "frank@logisticspro.ng", "enterprise", "2023-09-14", "EMEA"),
        ("Grace Kim", "grace@financeplus.kr", "pro", "2024-04-05", "APAC"),
        ("Henry Brown", "henry@mediahub.co.uk", "free", "2024-07-20", "EU-West"),
        ("Irene Novak", "irene@healthtech.cz", "enterprise", "2024-01-30", "EU-East"),
        ("James Wilson", "james@edusync.com", "pro", "2024-05-18", "US-East"),
    ]
    cur.executemany(
        "INSERT INTO customers (name, email, plan, signup_date, region) VALUES (?, ?, ?, ?, ?)",
        customers,
    )

    # ── Seed orders ────────────────────────────────────────────────
    products = [
        ("Cloud Platform License", 2999.00),
        ("Analytics Dashboard Pro", 499.00),
        ("API Gateway Subscription", 199.00),
        ("Data Warehouse Module", 1499.00),
        ("Security Compliance Pack", 799.00),
        ("AI/ML Toolkit", 999.00),
        ("DevOps Pipeline Suite", 349.00),
    ]
    statuses = ["completed", "completed", "completed", "shipped", "pending", "refunded"]
    random.seed(42)
    base_date = datetime(2024, 1, 1)

    orders = []
    for _ in range(30):
        cust_id = random.randint(1, 10)
        product, base_price = random.choice(products)
        amount = round(base_price * random.uniform(0.8, 1.2), 2)
        status = random.choice(statuses)
        order_date = base_date + timedelta(days=random.randint(0, 365))
        delivery_date = (
            (order_date + timedelta(days=random.randint(1, 14))).strftime("%Y-%m-%d")
            if status in ("completed", "shipped")
            else None
        )
        orders.append(
            (cust_id, product, amount, status, order_date.strftime("%Y-%m-%d"), delivery_date)
        )

    cur.executemany(
        "INSERT INTO orders (customer_id, product, amount, status, order_date, delivery_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        orders,
    )

    # ── Seed support tickets ───────────────────────────────────────
    ticket_subjects = [
        ("Login failure after password reset", "Customer cannot log in after resetting password. Error code AUTH-403."),
        ("Billing discrepancy on invoice", "Monthly invoice shows double charge for Analytics Dashboard Pro."),
        ("API rate limiting exceeded", "Production API calls returning 429 errors despite enterprise tier."),
        ("Data export timeout", "CSV export of 500K+ records hangs after 60 seconds."),
        ("SSO integration broken", "SAML SSO redirects fail with 'invalid assertion' after IdP certificate rotation."),
        ("Dashboard widgets not loading", "Several dashboard widgets show spinner indefinitely since last update."),
        ("Compliance report missing data", "SOC2 compliance report omits Q3 audit log entries."),
        ("Webhook delivery failures", "Outbound webhooks returning 502 for the past 12 hours."),
        ("Account downgrade request", "Customer wants to switch from enterprise to pro plan mid-cycle."),
        ("Performance degradation", "Query response times increased 3x after latest database migration."),
    ]
    priorities = ["critical", "high", "medium", "low"]
    sla_hours = {"critical": 4, "high": 8, "medium": 24, "low": 72}
    ticket_statuses = ["open", "open", "open", "in_progress", "in_progress", "escalated", "resolved"]

    tickets = []
    for i, (subject, description) in enumerate(ticket_subjects):
        cust_id = (i % 10) + 1
        priority = priorities[i % 4]
        status = random.choice(ticket_statuses)
        created = base_date + timedelta(days=random.randint(200, 365), hours=random.randint(0, 23))
        sla_deadline = created + timedelta(hours=sla_hours[priority])
        resolved_at = (
            (created + timedelta(hours=random.randint(1, sla_hours[priority] + 10))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if status == "resolved"
            else None
        )
        tickets.append(
            (
                cust_id,
                subject,
                description,
                priority,
                status,
                sla_deadline.strftime("%Y-%m-%d %H:%M:%S"),
                created.strftime("%Y-%m-%d %H:%M:%S"),
                resolved_at,
            )
        )

    # Add a few SLA-breached tickets explicitly
    tickets.append(
        (
            1, "URGENT: Production outage", "Full service outage affecting all enterprise customers",
            "critical", "escalated",
            "2024-09-01 04:00:00", "2024-09-01 00:00:00", None,
        )
    )
    tickets.append(
        (
            3, "Data loss incident", "Customer reports missing records in data warehouse after migration",
            "critical", "open",
            "2024-10-15 08:00:00", "2024-10-15 04:00:00", None,
        )
    )

    cur.executemany(
        "INSERT INTO support_tickets "
        "(customer_id, subject, description, priority, status, sla_deadline, created_at, resolved_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        tickets,
    )

    conn.commit()
    conn.close()
    print(f"✅ Database seeded at {DB_PATH}")
    print(f"   - 10 customers, {len(orders)} orders, {len(tickets)} support tickets")


def seed_policy_docs():
    """Create sample SOP/policy markdown documents."""
    os.makedirs(POLICIES_DIR, exist_ok=True)

    policies = {
        "sla_policy.md": """# Service Level Agreement (SLA) Policy

## Response Time Commitments

| Priority | First Response | Resolution Target | Escalation Trigger |
|----------|---------------|-------------------|--------------------|
| Critical | 30 minutes | 4 hours | After 2 hours |
| High | 2 hours | 8 hours | After 4 hours |
| Medium | 8 hours | 24 hours | After 16 hours |
| Low | 24 hours | 72 hours | After 48 hours |

## SLA Breach Procedures

1. **Automatic Escalation**: When a ticket exceeds 50% of its resolution target without update, it is auto-escalated to the next tier.
2. **Customer Notification**: SLA breach triggers an automatic apology email with a 10% service credit offer.
3. **Management Review**: All critical SLA breaches must be reviewed in the weekly ops meeting.
4. **Root Cause Analysis**: Critical and High breaches require a post-incident report within 48 hours.

## Compensation Guidelines

- **Single breach**: 10% credit on next invoice
- **Repeated breaches (3+ in 30 days)**: 25% credit + dedicated account manager assignment
- **Critical outage > 4 hours**: Full month credit + executive escalation
""",
        "refund_policy.md": """# Refund & Cancellation Policy

## Eligibility

- Full refund within **30 days** of purchase for annual plans.
- Pro-rated refund for mid-cycle cancellations on monthly plans.
- No refund for consumed API credits or one-time setup fees.

## Process

1. Customer submits refund request via support ticket or account settings.
2. Finance team reviews within **2 business days**.
3. Approved refunds processed to original payment method within **5-7 business days**.
4. Enterprise contracts: Refunds follow the Master Service Agreement (MSA) terms.

## Special Cases

- **Billing errors**: Immediate credit applied, no waiting period.
- **Service outage compensation**: Handled under SLA policy, not refund policy.
- **Plan downgrades**: Pro-rated credit applied to remaining billing cycle.
""",
        "escalation_procedures.md": """# Escalation Procedures

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
""",
        "data_handling_policy.md": """# Data Handling & Privacy Policy

## Data Classification

| Level | Examples | Access Control |
|-------|----------|---------------|
| Public | Marketing materials, docs | Open |
| Internal | Revenue reports, roadmaps | Employees only |
| Confidential | Customer PII, credentials | Role-based, encrypted |
| Restricted | Payment data, auth tokens | PCI-DSS compliant, need-to-know |

## Data Retention

- **Active customer data**: Retained while account is active + 90 days post-cancellation.
- **Support ticket history**: 3 years for compliance.
- **Audit logs**: 7 years minimum (SOC2/ISO27001 requirement).
- **Deleted data**: Hard-delete from backups within 30 days of request.

## Agent/Automation Rules

- Automated agents **MUST NOT** access Restricted data without explicit authorization token.
- All database queries from agents must be **read-only** unless via approved action ticket workflow.
- PII fields must be **masked** in logs and agent responses (show only last 4 chars of email/phone).
- Export operations > 10,000 records require **manager approval**.
""",
    }

    for filename, content in policies.items():
        filepath = os.path.join(POLICIES_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

    print(f"✅ Policy documents created in {POLICIES_DIR}")
    print(f"   - {len(policies)} policy files")


if __name__ == "__main__":
    seed_database()
    seed_policy_docs()
    print("\n🎉 All seed data initialized successfully!")

