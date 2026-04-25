# Case Study: hard_011 (Investor Dinner Cascade)

## Baseline (immediate submit)
- Reward: 0.5000
- Steps: 1
- Violations: 0
- Feedback: [constraints] 0/6 constraints met | [conflicts] No calendar conflicts | [commitments] No commitments created | [communication] MISSING email to Team | MISSING email to VP_Chen | [efficiency] 1 steps (optimal: 7)

## Improved policy
- Reward: 0.9900
- Steps: 5
- Violations: 0
- Feedback: [constraints] 6/6 constraints met | [conflicts] No calendar conflicts | [commitments] 1 honored | [communication] Email to Team: full credit | Email to VP_Chen: full credit | [efficiency] 5 steps (optimal: 7)

## Why improved policy scores higher
- Resolves lower-priority personal conflict (`cancel_event evt_90`)
- Preserves high-priority investor objective (`book_restaurant Sky Lounge`)
- Renegotiates existing social commitment via communication (`send_email Team`)
- Confirms delivery to executive stakeholder (`send_email VP_Chen`)
