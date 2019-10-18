import numpy as np
from scipy.spatial import distance_matrix

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [0] * len(self.entities)
        # apply agent physical controls
        p_force, k_force, has_kick = self.apply_action_force(p_force)
        # apply environment forces
        #p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force, k_force, has_kick)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

        self.collision_res()

    # gather agent action forces
    def apply_action_force(self, p_force):
        k_force = np.zeros((self.dim_p,))
        has_kick = False
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
                if agent.adversary:
                     p_force[i][0] = -p_force[i][0] 

            if hasattr(agent, 'kicking') and agent.kicking:
                if np.sum(np.square(self.landmarks[0].state.p_pos - agent.state.p_pos)) < np.square(self.landmarks[0].size + agent.size):
                    kick  = agent.action.kick.copy()
                    if agent.adversary: 
                        kick[0] = -kick[0]
                    k_force += kick
                    has_kick = True

        return p_force, k_force, has_kick
    
    def collision_res(self):
        s = [e.state for e in self.entities]
        p = np.array([ss.p_pos for ss in s])
        v = np.array([ss.p_vel for ss in s])
        # we assume the radius is the same for everything now. i don't care.
        r = self.entities[0].size
        p_old = p.copy()
        v_old = v.copy()
        d = distance_matrix(p, p) 
        d += np.identity(d.shape[0])*3*r
        coll_all = d < 2*r
        if np.any(coll_all):
            for i in range(len(s)):
                coll = coll_all[i]
                if np.any(coll):
                    delta = p_old-p_old[i]
                    c = 0.5 * (p_old[i] + p_old)
                    dnorm = delta/d[i][:,None]
                    p1 = c + r*dnorm
                    v_proj = np.einsum('ij,ij->i', v_old, dnorm)[:,None] * dnorm
                    v_proj_i = np.einsum('ij,ij->i', v_old[i,None], dnorm)[:,None] * dnorm
                    v1 = (v_old - v_proj) + v_proj_i
                    #print('\n\ni:',i,'\n proj:', v_proj, '\nv: ', v,'\n v1:', v1, '\n diff:', v_old - v_proj,'\n proj_i ', v_proj_i, '\ndnorm,',dnorm)
                    p = np.where(coll[:, None], p1, p)
                    v = np.where(coll[:, None], v1, v)
                    
            #print('equality:',np.equal(v_old,v))
        # abusing p_old insead of copying p to it again.
        np.clip(p[:,1],-0.5+r,0.5-r,p_old[:,1])  # y-walls
        np.clip(p[:-1,0],-1+r, 1-r,p_old[:-1,0]) # x-walls
        if abs(p[-1,1]) > self.goal_width-r: # ball x-wall
            p_old[-1, 0] = np.clip(p[-1,0],-1+r, 1-r)
        else:
            p_old[-1, 0] = p[-1, 0]

        # if the clipping changes the y coordinate we should flip the velocity vertically.
        # and same thing for the x coord.
        v[:, 0] = np.where(p[:,0] == p_old[:, 0], v[:,0], -v[:,0])
        v[:, 1] = np.where(p[:,1] == p_old[:, 1], v[:,1], -v[:,1])

        for i, s in enumerate(self.entities):
            s.state.p_pos = p_old[i]
            s.state.p_vel = v[i]

        

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):

        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            
            if hasattr(self, 'use_walls') and self.use_walls:
                is_ball = hasattr(entity_a, 'is_ball') and entity_a.is_ball

                if not (is_ball and abs(entity_a.state.p_pos[1]) < self.goal_width):
                    #can be sped up a bunch.
                    wall_dirs = [np.array([0,1]),np.array([0,-1]),np.array([1,0]),np.array([-1,0])]
                    wall_dists = [0.5, 0.5, 1, 1]
                    for i, w in enumerate(wall_dirs):
                        wall_thickness = 0.5
                        delta_pos = entity_a.state.p_pos - w * (wall_dists[i]+wall_thickness)
                        delta_pos = np.dot(delta_pos,w)*w
                        dist_min = entity_a.size + wall_thickness
                        force = self.force_from_delta_pos_and_min_dist(delta_pos, dist_min)
                        p_force[a] += force
        return p_force

    # integrate physical state
    def integrate_state(self, p_force, k_force, has_kick):
        for i,entity in enumerate(self.entities):
            is_ball = hasattr(entity, 'is_ball') and entity.is_ball
            if is_ball and has_kick:
                entity.state.p_vel = k_force

            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
                                                                  
            entity.state.p_pos += entity.state.p_vel * self.dt



    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    def force_from_delta_pos_and_min_dist_old(self, delta_pos, dist_min):
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        return force; 

    def force_from_delta_pos_and_min_dist(self, delta_pos, dist_min):
        dist_sq = np.dot(delta_pos, delta_pos)
        min_dist_sq = dist_min*dist_min
        diff = dist_sq - min_dist_sq
        force = 0
        if diff < 0:
            force = delta_pos*2.
        return force; 



    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist_min = entity_a.size + entity_b.size
        force = self.force_from_delta_pos_and_min_dist(delta_pos,dist_min)
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

