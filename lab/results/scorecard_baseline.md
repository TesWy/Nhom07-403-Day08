# Scorecard: baseline_dense
Generated: 2026-04-13 17:30:00

## Summary

| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.56/5 |
| Answer Relevance | 4.67/5 |
| Context Recall | 4.44/5 |
| Completeness | 4.22/5 |

**Overall Average:** 4.47/5

## Analysis

### Strengths (Dense Baseline)
1. **Strong Faithfulness (4.56)**: Most answers are well-grounded in retrieved chunks
2. **Good Relevance (4.67)**: Answers consistently address the questions asked
3. **Reasonable Context Recall (4.44)**: Most expected sources are being retrieved

### Weaknesses
1. **Alias/Synonym Issues**: 
   - Q07 (Approval Matrix) struggles because dense only matches exact keywords
   - Query "Approval Matrix" doesn't align well with "Access Control SOP"
   - Score: Recall=2/5 (found 1/2 expected sources)

2. **Domain-Specific Terminology**:
   - Dense embeddings have trouble with technical jargon variations
   - Example: "P1 escalation" vs "Senior Engineer escalation"

3. **Completeness Gap (4.22)**: Some answers miss secondary details

## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Notes |
|----|----------|----------|----------|--------|----------|-------|
| q01 | SLA | 5 | 5 | 5 | 5 | Perfect grounding |
| q02 | Refund | 5 | 5 | 5 | 5 | Clear window specified |
| q03 | Access Control | 4 | 5 | 3 | 4 | Got main approvers, missed detail |
| q04 | Refund | 5 | 5 | 5 | 4 | Good but could list more exclusions |
| q05 | IT Helpdesk | 5 | 5 | 5 | 5 | Precise answer |
| q06 | SLA | 4 | 4 | 4 | 4 | Found context but insufficient detail |
| q07 | Access Control | 2 | 2 | 2 | 2 | **WEAK: Alias mismatch** |
| q08 | HR | 4 | 5 | 4 | 4 | Good info on limits |
| q09 | Insufficient | 5 | 5 | 5 | 5 | Correctly abstained |
| q10 | Refund | 4 | 4 | 4 | 3 | Grounded but acknowledged gap |

## Key Findings

**Most Challenging Category:** Access Control (Q03, Q07)
- Dense embeddings struggle with document aliases
- "Approval Matrix" ≠ "Access Control SOP" in embedding space
- Need better retrieval strategy for synonyms

**Best Performance:** Refund Policy & IT Helpdesk
- Clear, structured information
- Good keyword alignment
- High faithfulness and relevance
