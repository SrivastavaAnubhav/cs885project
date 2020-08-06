from wu import Wu
from antq import AntQ
from gym import logger

logger.set_level(logger.DISABLED)

for i in range(5):
    # w = Wu("HHHHHPPHHHHPHH")
    # w.train()

    antq = AntQ("HHHHHPPHHHHPHH")
    antq.train()
