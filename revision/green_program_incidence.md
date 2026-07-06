# Green reskilling — who does it actually help?

**Question.** Green's inward reach is 22.8% and outward 54.5%, but the average worker
stays below one transition, so only a subset benefits. Are green's *successful switchers*
(workers who unlock ≥1 transition under green) concentrated in / near the genuinely-green
occupations, or spread across the workforce?

**Data.** `reskill-green_wage-opt_regC_2023` pickles, both flows. A "successful switcher"
= worker-group with `transition_viable` True at any step (COEFFY-weighted). Greenness of an
origin occupation = share of its linked ESCO skills (essential *or* optional) that are
green-tagged (`rp.green_skills` × `rp.occ_skills_mat_3d`). Reproduced by the notebook cell
that writes `results/tables/reskilling_simulation/{flow}/{flow}_green_incidence_by_origin_regC.csv`.

---

## Headline answer

Green reskilling does **not** preferentially help the green-adjacent workforce. Its
successful switchers are **no greener than the pool they came from** (mean green-skill
share: outward 7.3% vs pool 7.2%; inward 3.6% vs pool 3.4%), and origin greenness does
**not** predict who succeeds (outward Pearson r = **−0.14**). Instead green helps a **narrow,
skill-central technical slice** — engineering / electrotechnical / managerial-supervisory /
assembler / metal-working roles in manufacturing & construction (outward), and
transport-logistics clerks + machinery mechanics (inward) — largely regardless of how green
those occupations are.

A structural reason: green skills are almost never *essential* — only **4 of 125** ISCO-3
occupations have any green essential skill; green skills sit as *optional* (median green-skill
share 2.8%). So the green skills a worker adds are peripheral (low coreness), and who unlocks
a transition is set by the origin occupation's baseline skill connectivity, not its green
content.

---

## Outward (at-risk), green reach 54.5%

Switchers come from just **18 origin occupations**; top 4 hold 50%, top 9 hold 80%.

| origin occupation (ISCO-3) | % of switchers | success rate | green-skill share |
|---|---:|---:|---:|
| Engineering professionals (excl. electrotech) | 19.0 | 63.0% | 13.3% |
| Machinery mechanics & repairers | 12.8 | 58.1% | 3.1% |
| Manufacturing/mining/construction managers | 10.9 | 63.1% | 8.6% |
| Mining/manufacturing/construction supervisors | 9.5 | 56.8% | 5.7% |
| Mobile plant operators | 7.2 | 43.8% | 6.6% |
| Material-recording & transport clerks | 6.6 | 39.4% | 4.0% |
| Sheet/structural metal workers, welders | 5.6 | 58.5% | 1.6% |
| Electrical equipment installers & repairers | 4.6 | 43.6% | 9.1% |
| Building finishers & related trades | 4.3 | 44.2% | 12.8% |

Note the non-relationship: **Assemblers** (3.0% green) succeed at **70%**, while the
*greenest* origins — Process-control technicians (16.8% green), Building finishers (12.8%),
Physical/earth-science professionals (10.2%) — sit **below** average at 48%/44%/48%.
Sectors: manufacturing (C) 40%, construction (F) 28% (73% success), professional/scientific
(M) 8% (95% success).

## Inward (shortage), green reach 22.8%

Even narrower — only **4 origin occupations** ever switch; two hold 80%.

| origin occupation (ISCO-3) | % of switchers | success rate | green-skill share |
|---|---:|---:|---:|
| Material-recording & transport clerks | 59.2 | 27.7% | 4.0% |
| Machinery mechanics & repairers | 36.2 | 22.4% | 3.1% |
| Handicraft workers | 3.1 | 9.9% | 1.8% |
| Textile/fur/leather machine operators | 1.5 | 4.8% | 1.9% |

Sectors: transport & storage (H) 42% (67% success), wholesale/retail (G) 27%,
manufacturing (C) 12%. (The inward greenness↔success correlation is meaningless — 4 points
spanning a 1.8–4.0% greenness range.)

---

## Bottom line for the SI

It is neither "only the already-green" nor "a broad group" — green reskilling helps **the
skill-central technical trades**, which happen *not* to be the greenest. Greenness of the
origin occupation does not predict who benefits.
