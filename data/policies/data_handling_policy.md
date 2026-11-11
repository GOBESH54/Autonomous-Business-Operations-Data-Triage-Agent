# Data Handling & Privacy Policy

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
