from enum import IntEnum

class Role(IntEnum):
    IO   = 0      # 直接生成答案 / 推理
    REV  = 1      # Reviewer（检查＋精炼）
    # 以后扩展：  PLANNER,FUSION

NUM_ROLES = len(Role)