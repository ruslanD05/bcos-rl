# RQ5: Wall-colour generalisation

This directory stores evaluation-only results for RQ3 policies under held-out wall colours. Policies were trained on standard grey DoorKey walls and evaluated with purple, blue, red, and yellow walls while the layout, transition rules, and rewards remained unchanged.

The retained outputs are:

- `rq3_wall_ood_full/`: the original five-seed normal-versus-purple evaluation;
- `rq3_three_more_wall_colors/`: the five-seed extension to blue, red, and yellow walls.

These results test whether explanation alignment on standard training observations is associated with robustness to unseen wall colours. They do not modify wall colours during training.
