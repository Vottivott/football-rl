import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, team_size = 1):
        world = World()
        world.use_walls = True
        world.goal_width = 0.20
        # set any world properties first
        world.dim_c = 2
        num_good_agents = team_size
        num_adversaries = team_size
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.06
            agent.accel = 2.0
            agent.kicking = True

            angle = np.linspace(0, 2*np.pi, 16, endpoint=False)[np.newaxis,:]
            agent.kicks = np.concatenate([np.cos(angle), np.sin(angle)]).T * 2.35
            agent.discrete_action_space = False
            agent.max_speed = 0.75
            agent.is_ball = False

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        landmark = world.landmarks[0]
        landmark.name = 'landmark %d' % i
        landmark.collide = True
        landmark.movable = True
        landmark.size = 0.06
        landmark.is_ball = True
        landmark.adversary = -1

        # make initial conditions
        def done(agent, world):
            return self.is_goal(world) 

        world.is_scenareo_over = done
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
            agent.state.p_pos= [np.random.uniform(-(1-agent.size), +(1-agent.size), 1)[0],
                                np.random.uniform(-(0.5-agent.size), +(0.5-agent.size), 1)[0]]
            #Deterministic positions for debugging:
            #if agent.adversary:
            #    agent.state.p_pos = np.array([1.0, 0.5])
            #else:
            #    agent.state.p_pos = np.array([-1.0, -0.5])

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if True:#not landmark.boundary:
                landmark.state.p_pos = np.zeros(world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.is_movable = True # make the ball movable again.


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

    def is_goal(self, world):
        return abs(world.landmarks[0].state.p_pos[0]) > 1-world.landmarks[0].size and abs(world.landmarks[0].state.p_pos[1]) < world.goal_width - world.landmarks[0].size

    def agent_reward(self, world):
        goal = self.is_goal(world)
        r = 0
        if goal:
            # very large reward as to make sure that it's better to sometimes score than always end up in the corner 
            # not sure if it's important.
            r += 100 if world.landmarks[0].state.p_pos[0] > 0 else -100
        return world.landmarks[0].state.p_vel[0] + r

    def ray_circle_intersection(p, ray, rad):
        p_proj_dist = np.dot(p, ray)
        if p_proj_dist < 0:
            return False #we should still return and reuse this for the opposite ray. But what evs.
        d2 = np.dot(p, p) - p_proj_dist * p_proj_dist
        if d2 > rad:
             return False
        thc = np.sqrt(rad - d2)
        return p_proj_dist - thc

    def wall_intersect(v, d):
        return d/v

    def to_polar_form(p, v):
        dist = np.sqrt(np.dot(p, p))
        dir = p / dist
        delta_d = np.dot(v, dir)
        v_orth = v - delta_d * v 
        delta_theta = v_orth * dist 
        return delta_d, delta_theta

    #hmm if we do this we do this the teamsize is not important. We can train them first on small teams and then increase size!
    def observation_radar(self, agent, world):
        num_radars = 16
        angles = np.linspace(0, 2*np.pi, num_radars, endpoint=False)[np.newaxis,:]
        rays   = np.concatenate([np.cos(angle), np.sin(angle)]).T
        pos = agent.state.p_pos
        vel = agent.state.p_vel
        #distance, velocity in polar form, if it's a team_mate or not, is it a y-wall, is it a x-wall, if it's a ball
        sights[num_radars] = [float("inf"),0,0, 0, 0, 0] * num_radars
        for other in world.entities:
            if other is agent: continue
            p = other.state.p_pos-pos
            v = other.state.p_vel   #should this be relative as well?? For shooting probably not, for moving probably..

            if agent.adversary:
                p[0] *= -1
                v[0] *= -1

            for i in range(len(rays)):
                distance = ray_circle_intersection(p,rays[i], other.size)
                if sights[i][0] > distance:
                    sights[i] = [distance, *to_polar_form(v,p), other.adversary == agent.adversary, 0, 0, other.is_ball]

        for i in range(len(rays)):
            distance = wall_intersect(0.5-pos[1], rays[i][1])
            if sights[i][0] > distance:
                sights[i] = [distance, 0, 0, 0, 1, 0]
            distance = wall_intersect(-0.5-pos[1], rays[i][1])
            if sights[i][0] > distance:
                sights[i] = [distance, 0, 0, 0, 1, 0]
            distance = wall_intersect(1-pos[0], rays[i][0])
            if sights[i][0] > distance:
                sights[i] = [distance, 0, 0, 0, 1, 1]
            distance = wall_intersect(-1-pos[0], rays[i][0])
            if sights[i][0] > distance:
                sights[i] = [distance, 0, 0, 0, 1, 1]
        
        if agent.adversary: 
            pos[0] *= -1
            vel[0] *= -1
        
        return np.concatenate(sights + vel + pos)
            



    def observation(self, agent, world):
        team_mate_pos = []
        team_mate_vel = []
        opponent_pos = []
        opponent_vel = []

        ball_pos = world.landmarks[0].state.p_pos.copy()
        ball_vel = world.landmarks[0].state.p_vel.copy()

        agent_vel = agent.state.p_vel.copy()
        agent_pos = agent.state.p_pos-ball_pos

        for other in world.agents:
            if other is agent: continue
            p = other.state.p_pos-ball_pos
            v = other.state.p_vel.copy()
            if agent.adversary:
                p[0] *= -1
                v[0] *= -1
            
            if other.adversary == agent.adversary:
                team_mate_pos.append(p)
                team_mate_vel.append(v)
            else:
                opponent_pos.append(p)
                opponent_vel.append(v)

        if agent.adversary: # flip all them x-axis.
            agent_pos[0] *= -1
            agent_vel[0] *= -1
            ball_vel[0]  *= -1
            ball_pos[0]  *= -1
                
        #print(np.concatenate([agent_vel] + [agent_pos] + [ball_pos] + [ball_vel] + team_mate_pos + team_mate_vel + opponent_pos + opponent_vel))
        return np.concatenate([agent_vel] + [agent_pos] + [ball_pos] + [ball_vel] + team_mate_pos + team_mate_vel + opponent_pos + opponent_vel)
