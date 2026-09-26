# AGENTS.md — Gothenburg Tech Week × Chalmers Hackathon 2026 · SKF track

Goal: win the SKF track. Optimize for a clear problem → evidence → prototype → pitch story, not feature count.

## Event facts (Participant Schedule doc)

- Fri 25 – Sat 26 September 2026. Venue: FUSE, Chalmers, Gothenburg. Organizers: GTW × Chalmers.
- Tracks: NVIDIA, Saab, SKF, Ambidex, Open. We are SKF. Teams of 3–5, assignments fixed.
- Comms: Builderbase (Resources + Help Desk). Contact: isabella@gbgtechweek.com.
- Friday: 16:30 check-in (ready by 17:15) · 17:15–17:30 welcome · 17:30–~19:05 case presentations · then food + hacking, case-provider Q&A (venue open ≥21:00).
- Saturday: 09:00 breakfast + hacking · 12:00–13:00 rolling lunch · 14:00 pitch workshop (Joar & Jarryd) · 16:00–17:00 pitch prep (be in pitch room before 17:00) · 17:00–17:50 track pitches, 5 min/team, parallel tracks · 17:50–18:10 jury · 18:10 winners · 18:15–18:40 winning pitches · 18:40–18:50 closing · 19:00 end.
- Judging areas: relevance and problem understanding · innovation · feasibility · user value and impact · technical quality / prototype · scalability and next steps · presentation clarity.
- After: strongest solutions present at Tech Blueprint, 13 Oct 2026, World of Volvo.

## SKF challenge brief (track provider doc)

- Title: Analyze and compare learning methods suitable for industrial humanoids.
- Contacts: Meneka Karehalli (Process Specialist – Machining & Cleanliness), Magnus Wahlgard (Senior Strategist, Digital manufacturing). Virtual support day 1 + on request. Also offer: consultations, data sheets, old simulation results.
- Core question: How can we make the training of humanoids in an industrial setting easier and more efficient?
- Investigate: manual programming · learning from demonstration / teleoperation · imitation learning · reinforcement learning · simulation-to-real transfer · human feedback · vision-language-action / foundation-model approaches.
- Compare every method on: data effort · training time · safety · repeatability · adaptability · computing needs · explainability · deployment maturity. Split locomotion vs manipulation vs complete industrial workflows. Name where hybrid approaches beat any single method.
- MANDATORY build: a pipeline implementing locomotion “walking up the stairs” OR “change of step length”, with full documentation.
- Deliverable route A (depth): one task, full pipeline ending at simulation — results + code · dynamic simulation · docs answering “if it works, why? / if not, why?” · proposed learning pipeline for one SKF use case (data collection, training, testing, success measures).
- Deliverable route B (breadth): pipeline structure + analysis of ≥2 tasks (locomotion obligatory, manipulation desired: pick-and-place, wipe table / clean whiteboard) — feasibility theory · how-to · benefits/limits/data/safety/readiness · proposed SKF pipeline as above.
- Success criteria: good reasoning and concise conclusions · visible effort to implement · solid working simulation testable in real life · business KPIs and values.
- Format: slide deck + documentation + simulation results with videos.
- Notes: Google Cloud (compute-optimized) GPU access may be provided · Unitree G1 humanoid + hands data as input (rl_control_routine). Opportunities after: mentorship, internship, thesis partner, pilot.

## Tooling & credits (Credits doc — agents never set up, request, or store keys; user owns all accounts)

- Lovable is headline sponsor and primary tool. Code `COMM-GOTH-T6VY`: Pro Plan 1 (100 credits), choose monthly, redeem by end of event, do not share outside event. Combine other AI tools/compute alongside Lovable, not instead of it.
- Gonka compute brokers need Base URL + Model ID + API key. Models: `MiniMaxAI/MiniMax-M2.7` (weakest, high capacity, simple high-volume tasks) · `deepseek-ai/DeepSeek-V4-Flash-0731` and `zai-org/GLM-5.3-Flash` (strong, coding/agents). Error 429 = network out of capacity → wait, retry, or switch model.
- One-per-team keys at the organizer help desk (not listed publicly): Dahl API key (100M tokens, https://inference.dahl.global) · Proxy by Gonka.gg promo (150M tokens) · JoinGonka referral promo (~50M tokens).
- Shared codes: `HACKATHON26` on gonka-api (150M tokens, first 200 registrations) · `GOTHENBURGTECH` on Hyperfusion ($50, 300 users). Bonus, no code: Gonka24 ($10), Gonkarouter ($20).

## The whole process (Crash Course doc)

4 moves, 9 steps. Start with problem + people, build smallest useful proof, test what matters, tell one clear story.

Hackathon rule: timebox every move. Protect Sat 14:00 workshop, 16:00–17:00 prep, 17:00 pitch. If behind, reduce scope — do not skip validation (07) or pitch (09).

The process is a loop, not a checklist. Follow the evidence and revisit earlier moves when needed.

## Move 1: Discover

Understand the case before choosing a solution.

### 01 Understand — start with the situation, not the technology
- Do: who is affected, what are they trying to achieve, what blocks them, why does it matter?
- Capture: `[Customer]` struggles to `[goal]` because `[obstacle]`, resulting in `[consequence]`.
- Checkpoint: can you explain the problem without naming a technology?

### 02 Empathize — investigate real experiences
- Do: ask about the last real occurrence. What did they do instead? Where did it get difficult?
- Capture: We learned `[evidence]`. We think `[interpretation]`. We still need to verify `[assumption]`.
- Checkpoint: can you describe the problem from the customer's perspective?

### 03 Define — turn learning into a shared direction
- Do: what outcome should change, what context/constraint matters, what does success look like?
- Capture: How might we help `[customer]` achieve `[outcome]` when `[constraint]`?
- Checkpoint: is everyone solving the same challenge?

## Move 2: Choose

Create options, then decide what you need to prove.

### 04 Create — one idea is a reaction, several create a choice
- Do: generate individually, then share and combine. Compare problem fit, customer value, feasibility, testability.
- Capture: We selected `[concept]` because `[evidence and criteria]`.
- Checkpoint: can you explain why this concept beat the alternatives?

### 05 De-risk — find the assumption that could sink the idea
- Do: list desirability, feasibility, viability assumptions. Pick the most critical and least supported.
- Capture: We believe `[assumption]`. If false, `[consequence]`. Test: `[experiment]`.
- Checkpoint: do you know what the prototype must help you learn?

## Move 3: Prove

Build what matters. Learn fast. Make it credible. The prototype is evidence, not the finish line.

### 06 Build — show the value, skip the feature buffet
- Do: define must-demonstrate / useful-if-time / not-now. Use the lightest testable prototype (mock-up, storyboard, PoC, simulated service is fine).
- Capture: Our prototype shows how `[customer]` uses `[solution]` to achieve `[outcome]`.
- Checkpoint: does every feature help prove the core value?

### 07 Test — "they liked it" is not enough
- Do: give someone a realistic scenario. Observe before helping. Capture confusion, evidence, changes.
- Capture: Worked: `[...]` | Confused: `[...]` | Evidence: `[...]` | Change: `[...]`
- Checkpoint: can you explain what changed because of testing?

## Move 4: Make it real

### 08 Make it real — what happens after the hackathon?
- Do: who will use, approve, fund, own, maintain it? What tech, data, partners are needed? What is the next experiment?
- Capture: Next: `[action]` | Owner/partner: `[who]` | Question answered: `[what]`
- Checkpoint: can you name one credible next step?

### 09 Pitch — problem, evidence, solution, proof, next step
- Do: problem + customer + evidence | solution + prototype + learning | next step + implementation.
- Capture: We help `[customer]` overcome `[problem]` through `[solution]`, enabling `[outcome]`.
- Checkpoint: can the jury follow the line from problem to evidence to solution?

## Judge-ready in 60 seconds

Answer all five, in order:
1. What problem matters?
2. What evidence supports it?
3. What did you build?
4. What did you learn?
5. What happens next?

## Instructions for agents in this repo

- Work move by move. State current step (01–09) and its checkpoint before advancing.
- Use the CAPTURE templates verbatim when recording decisions in docs.
- Never propose technology in 01–03. If a solution appears early, park it under 04 Create.
- Every build item must trace to the 05 riskiest assumption or the 06 must-demonstrate list, and serve the SKF mandatory core: locomotion pipeline (stairs OR step length) + full docs + simulation videos. Cut the rest.
- Prefer docs + lightest prototype that makes the core idea testable. Route A (depth) is the default unless evidence favors route B.
- When time is short, cut scope, never validation (07) or pitch (09).
- Never set up, request, or use API keys / credits. User owns Lovable + Gonka/Brev setup.
- Keep external references general: IDEO U Design Thinking overview (14 min), Stanford Design Thinking Bootleg, IDEO.org Field Guide.
