import argparse
import sys
#sys.path.appeng('../iGibsonChallenge2021/')

from simple_agent import RandomAgent, ForwardOnlyAgent

from gibson2.challenge.test_sn import Challenge


def get_agent(agent_class, ckpt_path=""):
    if agent_class == "Random":
        return RandomAgent()
    if agent_class == "ForwardOnly":
        return ForwardOnlyAgent()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-class", type=str, default="ForwardOnly", choices=["Random", "ForwardOnly", "SAC"])
    parser.add_argument("--ckpt-path", default="", type=str)

    args = parser.parse_args()

    agent = get_agent(
        agent_class=args.agent_class,
        ckpt_path=args.ckpt_path
    )
    challenge = Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()
