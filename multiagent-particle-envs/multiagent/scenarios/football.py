import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 2
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075
            agent.accel = 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        landmark = world.landmarks[0]
        landmark.name = 'landmark %d' % i
        landmark.collide = True
        landmark.movable = True
        landmark.size = 0.3#05
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if True:#not landmark.boundary:
                landmark.state.p_pos = np.zeros(world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.agent_reward(world) if agent.adversary else -self.agent_reward(world)
        return main_reward

    def agent_reward(self, world):
        return world.landmarks[0].state.p_vel[0];

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        team_mate_pos = []
        team_mate_vel = []
        opponent_pos = []
        opponent_vel = []
        for other in world.agents:
            if other is agent: continue
        if other.adversary == agent.adversary:
            team_mate_pos.append(other.state.p_pos-world.landmarks[0].state.p_pos);
            team_mate_vel.append(other.state.p_vel);
        else:
            opponent_pos.append(other.state.p_pos-world.landmarks[0].state.p_pos);#agent.state.p_pos);
            opponent_vel.append(other.state.p_vel);
        ball_position = world.landmarks[0].state.p_pos-agent.state.p_pos
        ball_vel = world.landmarks[0].state.p_vel
    #comm.append(other.state.c)
    #other_pos.append(other.state.p_pos - agent.state.p_pos)
    #if not other.adversary:
    #    other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [ball_position] + [ball_vel] + team_mate_pos + team_mate_vel + opponent_pos + opponent_vel)
