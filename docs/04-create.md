# 04 Create — concepts (parked here, per rule)

Candidates, compared on problem fit / documentability / feasibility overnight /
testability in simulation:

1. **Curriculum + domain-randomization ladder.** Flat → obstacles → low
   friction → straight stairs → stairs with turn. Gated promotion on success
   metrics. Directly answers the workshop context; trivially documentable.
2. **Classical baseline + RL residual.** A simple periodic gait / trajectory
   generator carries the robot; a small RL policy learns corrections for
   steps, slips and turns. Novel for the jury, explainable ("if it works,
   why?"), trains fast.
3. **Imitation bootstrap + RL fine-tune.** Seed from Unitree G1 reference
   motion (`rl_control_routine`), refine with RL on stairs. Strong if SKF
   hands over motion data; weak if they do not.
4. **VLA supervisor + locomotion skill.** Language-commanded task switching
   ("climb, avoid the spill"). Powerful story, infeasible to train overnight;
   keep as proposed-next-step only.

We selected the curriculum ladder (1) carrying an RL-residual policy (2)
because it is the only combination that is documentable stage-by-stage,
testable in simulation tomorrow, and explicitly encodes the providers'
workshop context as randomized axes. (3) is the fallback if G1 motion data
arrives; (4) is parked as the next step for the pitch.
