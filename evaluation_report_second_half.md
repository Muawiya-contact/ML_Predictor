# Second-Half Evaluation Report
Clinical-concept extraction of 20 Roman Urdu chief complaints through the offline Qwen2.5 pipeline, scored against gold English references.
- Generation model: `qwen2.5:latest` (local Ollama, temperature 0.0)
- Generated: 2026-09-20 22:26:00
- Scoring: symmetrical stop-word removal, underscore/slash token splitting, semantic-equivalence matching (e.g. tachycardia↔palpitations, vertigo↔dizziness), bag overlap.
- Per item: `P = N_match / N_gen`, `R = N_match / N_ref`, `F1 = 2PR/(P+R)`.

## Items 1-10
| SN | Roman Urdu Complaint | Model Concept (Qwen2.5) | Gold Reference | P | R | F1 |
|----|----------------------|--------------------------|---------------|-------|-------|-------|
| 1 | Chest pain aur ghabrahat khana khane ke baad se aur thakan bohat ho rahi hai. | chest pain anxiety eating palpitations | Chest pain and palpitations anxiety since after eating and extreme fatigue. | 1.0000 | 0.7143 | 0.8333 |
| 2 | Chest pain left side khana khane ke baad se aur pasina bohat aa raha hai. | chest pain sweating left side | Left-sided chest pain since after eating and excessive sweating. | 0.6000 | 0.5000 | 0.5455 |
| 3 | Tez chest pain aur saans phool rahi hai aadhay ghante se aur saans lene mein takleef hai. | severe chest pain dyspnea | Severe chest pain and shortness of breath for half an hour and difficulty breathing. | 0.7500 | 0.3333 | 0.4615 |
| 4 | Chest pain aur saans phool rahi hai do ghante se, haath thanday ho gaye hain. | chest pain dyspnea cold hands | Chest pain and shortness of breath for two hours, hands have become cold. | 0.8000 | 0.5000 | 0.6154 |
| 5 | Patient ko chest pain aur ghabrahat ho raha hai kal raat se aur ghabrahat bhi ho rahi hai. | chest pain anxiety | Patient has had chest pain and palpitations anxiety since last night. | 1.0000 | 0.5000 | 0.6667 |
| 6 | Achanak palpitations ek ghante se, known cardiac patient hai. | palpitations | Sudden palpitations for one hour, is a known cardiac patient. | 1.0000 | 0.2000 | 0.3333 |
| 7 | Bohat zyada dil tez dhadak raha hai khana khane ke baad se aur bohat kamzori lag rahi hai. | palpitations weakness | Heart is beating very fast since after eating and feeling very weak. | 0.0000 | 0.0000 | 0.0000 |
| 8 | Mamuli chest mein pressure aadhay ghante se aur ulti jaisa lag raha hai, family mein heart disease hai. | chest pressureodynalgia family history of heart disease | Mild pressure in the chest for half an hour and feeling nauseous, family history of heart disease. | 0.8000 | 0.4000 | 0.5333 |
| 9 | Seena jakar raha hai aadhay ghante se aur chakkar bhi aa rahe hain. | chest pain dizziness | Chest tightness for half an hour and also feeling dizzy. | 0.3333 | 0.1667 | 0.2222 |
| 10 | Shadeed seena jakar raha hai do ghante se. | severe chest pain | Severe chest tightness constriction for two hours. | 0.6667 | 0.4000 | 0.5000 |

## Items 11-20
| SN | Roman Urdu Complaint | Model Concept (Qwen2.5) | Gold Reference | P | R | F1 |
|----|----------------------|--------------------------|---------------|-------|-------|-------|
| 11 | Seena mein jalan ka ehsaas aram karte hue bhi aur dard peeth tak ja raha hai. | chest pain relieved intermittent | Feeling of burning in the chest even at rest and pain radiating to the back. | 0.5000 | 0.2857 | 0.3636 |
| 12 | Shadeed chest tightness sotay waqt achanak se aur thakan bohat ho rahi hai. | severe chest tightness palpitations | Severe chest tightness suddenly while sleeping and extreme fatigue. | 0.7500 | 0.4286 | 0.5455 |
| 13 | Halki left arm mein dard subah se aur thakan bohat ho rahi hai. | left arm pain sleep disturbance | Mild pain in the left arm since morning and extreme fatigue. | 0.6000 | 0.4286 | 0.5000 |
| 14 | Mamuli chest mein tez dard achanak se aur bohat kamzori lag rahi hai, family mein heart disease hai. | chest pain palpitations family history of heart disease | Mild chest with sudden sharp pain and feeling very weak, family history of heart disease. | 0.8333 | 0.5000 | 0.6250 |
| 15 | Seena bhaari lag raha hai achanak se, ghabrahat bhi ho rahi hai. | shortness of breath anxiety | Feeling of heaviness in the chest suddenly, along with anxiety palpitations. | 0.3333 | 0.1429 | 0.2000 |
| 16 | Halki chest tightness kaam karte waqt se aur pasina bohat aa raha hai, purani cardiac history hai. | chest tightness sweating cardiac history | Mild chest tightness while working and excessive sweating, old cardiac history. | 1.0000 | 0.5714 | 0.7273 |
| 17 | Back mein dard ke sath seena tight achanak se, haath thanday ho gaye hain. | back pain chest cold hands cold | Sudden chest tightness along with back pain, hands have become cold. | 0.8333 | 0.5556 | 0.6667 |
| 18 | Tez irregular heartbeat aadhay ghante se aur dard kandhay tak ja raha hai, diabetes ka mareez hai. | palpitations chest pain diabetes | Fast irregular heartbeat for half an hour and pain radiating to the shoulder, diabetic patient. | 0.2500 | 0.1111 | 0.1538 |
| 19 | Left arm mein dard sotay waqt achanak se aur ulti jaisa lag raha hai. | left arm pain sudden onsetodynema | Sudden pain in the left arm while sleeping and feeling nauseous. | 0.8000 | 0.5714 | 0.6667 |
| 20 | Tez seena mein dard aur pasina seedhiyan chadhte waqt se aur dard peeth tak ja raha hai. | severe chest pain sweating radiating to back | Severe chest pain and sweating while climbing stairs and pain radiating to the back. | 1.0000 | 0.6667 | 0.8000 |

## Aggregated Statistics

### Overall (n=20)
| Metric | N | Mean +/- SD |
|--------|---|-------------|
| Precision | 20 complaints | 0.6925 +/- 0.2777 |
| Recall | 20 complaints | 0.3988 +/- 0.1892 |
| F1 | 20 complaints | 0.4980 +/- 0.2185 |

### Food-related subset (items 1, 2, 7; n=3)
| Metric | N | Mean +/- SD |
|--------|---------------|-------------|
| Precision | 3 complaints | 0.5333 +/- 0.4110 |
| Recall | 3 complaints | 0.4048 +/- 0.2993 |
| F1 | 3 complaints | 0.4596 +/- 0.3456 |

## Key Observations
- F1 across the 20 complaints: min 0.000 (item 7), max 0.833 (item 1), median 0.545.
- Weakest items are 7, 18, 15, 9; their generated concepts recover the fewest gold tokens.
- Food-related subset (items 1, 2, 7): F1 0.460 +/- 0.346.
- Recall is systematically lower than precision: the model names each finding once, while gold references use fuller clinical phrasing (e.g. 'heart is beating very fast' vs 'palpitations').
