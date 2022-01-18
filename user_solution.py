# from policies.chair_push import ChairPolicy
# from policies.chair_push_new import ChairNewPolicy
# from policies.chair_push_nudge import ChairNudgePolicy
# from policies.drawer_open import DrawerPolicy
# from policies.door_open import DoorPolicy
# from policies.door_open_new import DoorNewPolicy
# from policies.door_open_drake import DoorDrakePolicy
# from policies.prehensile_bucket_move import PrehensileBucketPolicy
# from policies.nonprehensile_bucket_move import NonPrehensileBucketPolicy
from policies.hybrid_bucket_move import HybridBucketPolicy

# Change Accordingly [NonPrehensileBucketPolicy, PrehensileBucketPolicy, HybridBucketPolicy, DoorPolicy, ChairPolicy, DrawerPolicy]
UserPolicy = HybridBucketPolicy ## NonPrehensileBucketPolicy
