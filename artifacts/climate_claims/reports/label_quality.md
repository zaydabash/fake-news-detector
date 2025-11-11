# Label Quality Report: climate_claims

## Overview
- **Niche**: climate_claims
- **Total Samples**: 194
- **Generated**: 2025-10-10 21:41:26

## Label Distribution
- **claim**: 77 (39.7%)
- **scientific**: 117 (60.3%)
- **Unlabeled**: 0 (0.0%)

## Heuristic Coverage
- **Domain Matches**: 188 (96.9%)
- **Phrase Matches**: 33 (17.0%)
- **Conflicts**: 0 (0.0%)

## Configuration Used
- **Positive Phrases**: hoax, climate scam, ice age next year, cities underwater, climate lockdown...
- **Negative Phrases**: IPCC, radiative forcing, anomaly, confidence interval, peer-reviewed...
- **Positive Domains**: climatedepot.com, iceagenow.info, climatechangereconsidered.org, climate.news, notrickszone.com...
- **Negative Domains**: ipcc.ch, noaa.gov, climate.nasa.gov, nature.com, washingtonpost.com...

## Sample Data (First 10 Rows)
              id label               source                           heuristics                                                                                                text_preview
climate_claims_0 claim  wattsupwiththat.com        domain_type_fallback:positive “…the world’s most viewed climate website”\n\n– Fred Pearce The Climate Files:\n\nThe Battle for the Tru...
climate_claims_1 claim  wattsupwiththat.com        domain_type_fallback:positive   Pro: Many photos show the ugly pollution being emitted\n\nThe screen grab below is from this Washingto...
climate_claims_2 claim  wattsupwiththat.com        domain_type_fallback:positive   If you haven’t already, please read the submission guidelines.\n\nUse the form below to compose your s...
climate_claims_3 claim  wattsupwiththat.com        domain_type_fallback:positive   The 130-Degree F Reading in Death Valley Is A World Record\n\nCon: 134 degrees in 1913 is still the of...
climate_claims_4 claim  wattsupwiththat.com        domain_type_fallback:positive   Watts Up With That? – Climate Change News, Research & Analysis\n\nWelcome to Watts Up With That, one o...
climate_claims_5 claim  wattsupwiththat.com        domain_type_fallback:positive “…the world’s most viewed climate website”\n\n– Fred Pearce The Climate Files:\n\nThe Battle for the Tru...
climate_claims_6 claim www.climatedepot.com positive_domain:www.climatedepot.com     Off-the-charts insanity: “Svensmark recalls a [climate realist] conference in Germany at which he ga...
climate_claims_7 claim www.climatedepot.com positive_domain:www.climatedepot.com   By Samantha Chang\n\nDemocrats and their left-wing media lapdogs beclowned themselves after their manu...
climate_claims_8 claim     climatedepot.com     positive_domain:climatedepot.com     No mobile information will be shared with third parties/affiliates for marketing/promotional purpose...
climate_claims_9 claim www.climatedepot.com positive_domain:www.climatedepot.com   Labour peer Lord Brooke has offered his pearls of wisdom on the Assisted Dying Bill debate today:\n\n“...

## Manual Audit Checklist
- [ ] Review 100 random samples for label accuracy
- [ ] Check for systematic biases in domain-based labeling
- [ ] Verify phrase-based heuristics are appropriate
- [ ] Assess class balance and potential improvements
- [ ] Consider additional heuristics for unlabeled samples

## Limitations & Biases
- **Domain Bias**: Labels heavily influenced by source domain
- **Phrase Bias**: May miss nuanced content that doesn't contain key phrases
- **Temporal Bias**: Heuristics may not adapt to changing language patterns
- **Cultural Bias**: English-only phrases may miss non-English content
- **Context Bias**: Short texts may lack sufficient context for accurate labeling

## Recommendations
1. **Manual Review**: Audit at least 100 samples for each class
2. **Iterative Improvement**: Refine heuristics based on manual review
3. **Balance Check**: Ensure reasonable class distribution
4. **Quality Threshold**: Consider minimum sample requirements per class
5. **Validation Split**: Set aside data for manual validation
