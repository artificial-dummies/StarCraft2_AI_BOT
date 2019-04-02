import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2 import position
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, SCV, DRONE, ROBOTICSFACILITY, OBSERVER, FLEETBEACON, \
 ZEALOT, STALKER, CARRIER, ROBOTICSBAY, COLOSSUS, FORGE, TWILIGHTCOUNCIL, \
 PROTOSSGROUNDWEAPONSLEVEL1, PROTOSSGROUNDWEAPONSLEVEL2, PROTOSSGROUNDWEAPONSLEVEL3, \
 PROTOSSGROUNDARMORSLEVEL1, PROTOSSGROUNDARMORSLEVEL2, PROTOSSGROUNDARMORSLEVEL3, \
 PROTOSSSHIELDSLEVEL1, PROTOSSSHIELDSLEVEL2, PROTOSSSHIELDSLEVEL3, \
 PROTOSSAIRWEAPONSLEVEL1, PROTOSSAIRWEAPONSLEVEL2, PROTOSSAIRWEAPONSLEVEL3, \
 PROTOSSAIRARMORSLEVEL1, PROTOSSAIRARMORSLEVEL2, PROTOSSAIRARMORSLEVEL3
 
import random
import cv2
import numpy as np
import os
import time
import math
import keras

# os.environ["SC2PATH"] = 'D:/StarCraft II/'

HEADLESS = True

# Daniel PC command
# D:/Python/Python37-32/python.exe c:/Users/danie/Desktop/StarCraft2_AI_BOT/Bot/SC2_AI_BOT_AD.py

def get_session(gpu_fraction=0.3):
    """Assume that you have 6GB of GPU memory and want to allocate ~2GB"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())


class ADBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1):
        self.do_something_after = 0
        self.use_model = use_model
        self.title = title
        self.military_units = [ZEALOT, VOIDRAY, STALKER, CARRIER, COLOSSUS]
        self.choice_success = False

        # every iteration, make sure that unit id still exists!
        self.scouts_and_spots = {}

        # Choices that the bot can make #
        self.choices = {0: self.build_scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        5: self.build_worker,
                        6: self.build_assimilator,
                        7: self.build_airforce,
                        8: self.build_pylon,
                        9: self.defend_nexus,
                        10: self.attack,
                        11: self.expand,
                        12: self.do_nothing,
                        13: self.build_colossus,
                        14: self.build_carrier,
                        15: self.upgrade_attack,
                        16: self.upgrade_armor,
                        17: self.upgrade_shields,
                        18: self.upgrade_air_attack,
                        19: self.upgrade_air_armor
                        }
        # initializes the array holding the training data
        self.train_data = []

        # If useModel is set to true this will load up the model
        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("AD-100-epochs-0.001-LR-STAGE1.model")

    # when the game ends we store the training data, also if the model was used we store whether it won or not
    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        if game_result == Result.Victory:
            print(self.train_data)
            np.save("train_data/DV{}.npy".format(str(int(time.time()))), np.array(self.train_data))

        if self.use_model:
            with open("gameout-model-vs-easy.txt","a") as f:
                f.write("Model {} - {}\n".format(game_result, int(time.time())))
    
    # this happens in each iteration
    async def on_step(self, iteration):

        # print('Time:',self.time)

        # distribute workers every 5 seconds, don't want those workers doing nothing
        if iteration % 5 == 0:
            await self.distribute_workers()
        
        # send scouts to possible enemy locations
        await self.scout()

        # gets intel for and draws a picture in cv2 with the given intel
        await self.intel()

        # Lets the bot make a choice, if the model is being used it will use that model, else make a random choise of those 20 choices
        await self.do_something()

    # used to get a random location on a specific area when scouting with probes
    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        x += random.randrange(-5,5)
        y += random.randrange(-5,5)

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to

    # assigns a scout to a location
    async def scout(self):
        self.expand_dis_dir = {}

        # finds locations for sending scouts
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        # removes dead scouts
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        # until we have a robotics facility, we use probes to scout, else we use observerse (that are cloaked)
        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        # assigns a scout to scout, adds the scout to a location array to let know that he shouldn't be moved again
        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass

        # if we are scouting with a probe, move him around the scouting area so it is harder to kill
        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    # gets intel for the cv2 drawing, unit types and structures are drawn differently, as well as enemy units
    async def intel(self):

        # gets the map info for the drawing canvas
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # draws a circle for all of our units as a light gray color, their radius is the radius of the unit given by the game
        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        # does the same as drawing our units, exept these are enemy units. Drawn with a darker gray color
        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:
            # this try block gets information of the current state we care about in the game, that is how many minerals, vespene gas, population with a ratio...
            # ... towards our supply cap and the ratio of workers against the supplies we currently have
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            print(str(e))


        # flip horizontally to make our final fix in visual representation
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)

        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        # we can set headless to true or false, depending if we want to see the cv2 drawing while running the game
        if not HEADLESS:
            if self.use_model:
                cv2.imshow(str(self.title), resized)
                cv2.waitKey(1)
            else:
                cv2.imshow(str(self.title), resized)
                cv2.waitKey(1)

     ###################################################################################################################
    # All following build options always checks if the required structures are ready and there is no queue.             #
    # This is so the ai will not allocate unessisary recources, it can train as soon as the queue is empty.             #
    # Since the ai works so fast, it will always be quick enough to add another unit to train.                          #
    # There is also always a check that checks wether the bot can afford the unit/structure/upgrade its trying to do.   #
    # Buildings build locations are calculated by a random distance from an existing pylon within the psionic matrix.   #
    # Psionic matrix is a powered area created by pylons that the protoss can build their structures on.                #
     ###################################################################################################################

    # Tries to build a scout (observer), if it doesn't exist we try to build a robotics facility that can create observers
    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                self.choice_success = True
                await self.do(rf.train(OBSERVER))
                break
        if len(self.units(ROBOTICSFACILITY)) < len(self.units(NEXUS)):
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                    self.choice_success = True
                    await self.build(ROBOTICSFACILITY, target)

    # Builds probes (workers)
    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE):
                self.choice_success = True
                await self.do(random.choice(nexuses).train(PROBE))

    # builds zealots, earliest available unit for the protos, 
    async def build_zealot(self):
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                self.choice_success = True
                await self.do(random.choice(gateways).train(ZEALOT))

    # builds gateways, will not build more gateways than the number of nexuses times 2
    async def build_gateway(self):
        if len(self.units(GATEWAY)) < len(self.units(NEXUS).ready) * 2:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                self.choice_success = True
                await self.build(GATEWAY, target)

    # builds voidray, if a stargate doesn not exist we build the next increment in our airforce buildings
    async def build_voidray(self):
        stargates = self.units(STARGATE).ready.noqueue
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                self.choice_success = True
                await self.do(random.choice(stargates).train(VOIDRAY))
        #####
        else:
            await self.build_airforce()

    # builds stalkers, ground range unit, if it cannot it will build the required structure needed to build stalkers
    async def build_stalker(self):
        pylon = self.units(PYLON).ready.noqueue.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                self.choice_success = True
                await self.do(random.choice(gateways).train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                    self.choice_success = True
                    await self.build(CYBERNETICSCORE, target)
    
    # Builds a carrier, end game flying unit, if it cannot it will increment the next building in our airforce, if it can afford it
    async def build_carrier(self):
        stargates = self.units(STARGATE).ready
        fleetbeacons = self.units(FLEETBEACON)
        if stargates.exists and fleetbeacons.exists:
            if self.can_afford(CARRIER):
                self.choice_success = True
                await self.do(random.choice(stargates).train(CARRIER))
            else:
                await self.build_airforce()

    # Builds a colossus, end game ground unit, if it cannot it will build the required structure (robotics facility and robotics bay)
    async def build_colossus(self):
        robotics_facility = self.units(ROBOTICSFACILITY)
        robotics_bay = self.units(ROBOTICSBAY)
        if robotics_bay.exists and robotics_facility.exists:
            if self.can_afford(COLOSSUS):
                self.choice_success = True
                await self.do(random.choice(robotics_facility).train(COLOSSUS))
        else:
            self.choice_success = True
            await self.build_robotics_bay()

    # Tries to build a robotics bay
    async def build_robotics_bay(self):
        robotics_facility = self.units(ROBOTICSFACILITY)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if robotics_facility.ready.exists:
                if not self.units(ROBOTICSBAY).exists:
                    if self.can_afford(ROBOTICSBAY):
                        target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                        await self.build(ROBOTICSBAY, target)

    # Builds an assimilator on vespene geysers to collect vespene gas
    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    self.choice_success = True
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    # Builds a forge for upgrading units
    async def build_forge(self):
        if self.units(NEXUS).ready.exists and self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.can_afford(FORGE) and not self.already_pending(FORGE) and len(self.units(NEXUS)) > len(self.units(FORGE)) and len(self.known_enemy_units(FORGE)) < 2:
                target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                await self.build(FORGE, target)

    # builds a twilight council to get level 2 and 3 upgrades for ground units and shields  
    async def build_twilight_council(self):
        if self.units(CYBERNETICSCORE).ready.exists and self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            twilightcouncil = self.units(TWILIGHTCOUNCIL)
            if not twilightcouncil.exists:
                if self.can_afford(TWILIGHTCOUNCIL) and not self.already_pending(TWILIGHTCOUNCIL):
                    target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                    await self.build(TWILIGHTCOUNCIL, target)

    # builds the airforce hierarchy, from the required structures to start to end game structures
    async def build_airforce(self):
        cybernetics_cores = self.units(CYBERNETICSCORE)
        fleet_beacon = self.units(FLEETBEACON)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE) and len(self.units(STARGATE)) < len(self.units(NEXUS).ready):
                    target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                    self.choice_success = True
                    await self.build(STARGATE, target)
                elif not fleet_beacon.exists:
                    if self.units(STARGATE).ready.exists:
                        if self.can_afford(FLEETBEACON) and not self.already_pending(FLEETBEACON):
                            target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                            self.choice_success = True
                            await self.build(FLEETBEACON, target)

            if not cybernetics_cores.exists:
                if self.units(GATEWAY).ready.exists:
                    if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                        target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                        self.choice_success = True
                        await self.build(CYBERNETICSCORE, target)

    # builds a pylon. Pylonds are used to highten the supply left as well as creating a psionic matrix around them for buildings...
    # ... to be warped in.
    async def build_pylon(self):
        # added a random position with more distance to either pylons or nexuses
        nexuses = self.units(NEXUS).ready
        if nexuses.exists:
            if self.can_afford(PYLON) and not self.already_pending(PYLON) and self.supply_left < 5:
                if self.units(PYLON).ready.exists:
                    pylon = self.units(PYLON).ready.random
                    target1 = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 7))
                    target2 = await self.find_placement(PYLON, near=random.choice(self.units(NEXUS)).position.towards(self.game_info.map_center, 10))
                    choice = [target1, target2]
                    self.choice_success = True
                    await self.build(PYLON, random.choice(choice))
                else:
                    self.choice_success = True
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 7))

    # expands to another base location with minerals and vespene gas
    async def expand(self):
        try:
            if self.can_afford(NEXUS) and len(self.units(NEXUS)):
                self.choice_success = True
                await self.expand_now()
        except Exception as e:
            print(str(e))

    # sometimes it is good to do nothing, here the bot will do nothing and then wait a random time from 7 to 100 milliseconds
    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.choice_success = True
        self.do_something_after = self.time + wait

    # makes all units defend a nexus that are within a range of a nexus
    async def defend_nexus(self):
        if len(self.known_enemy_units) > 0:
            nexus = random.choice(self.units(NEXUS))
            target = self.known_enemy_units.closest_to(nexus)
            if nexus.distance_to(target) < 25:
                for unit_type in self.military_units:
                    for u in self.units(unit_type).idle: 
                        self.choice_success = True
                        await self.do(u.attack(target))

    # attacks known enemy units, the known_enemy_units function also returns known enemy structures
    async def attack(self):
        for unit_type in self.military_units:
            if len(self.units(unit_type)) > 4:
                if len(self.known_enemy_units) > 0:
                    self.choice_success = True
                    await self.attack_known_enemy_unit()
    
    # helper function for the attack
    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            for unit_type in self.military_units:
                for u in self.units(unit_type).idle:
                    target = random.choice(self.known_enemy_units).position
                    await self.do(u.scan_move(target))

    # Upgrades attack for all ground units, from level 1 to 3, builds the required structures if they do not exist
    async def upgrade_attack(self):
        if self.units(FORGE).ready.exists:
            forge = self.units(FORGE).ready.noqueue
            if forge.exists and self.already_pending_upgrade(PROTOSSGROUNDWEAPONSLEVEL1) == 0:
                if self.can_afford(PROTOSSGROUNDWEAPONSLEVEL1):
                    self.choice_success = True
                    await self.do(forge.first.research(PROTOSSGROUNDWEAPONSLEVEL1))
            elif self.already_pending_upgrade(PROTOSSGROUNDWEAPONSLEVEL2) == 0 and self.already_pending_upgrade(PROTOSSGROUNDWEAPONSLEVEL1) == 1:
                if self.units(TWILIGHTCOUNCIL).ready:
                    if self.can_afford(PROTOSSGROUNDWEAPONSLEVEL2):                    
                        self.choice_success = True
                        await self.do(forge.first.research(PROTOSSGROUNDWEAPONSLEVEL2))
                else:
                    self.choice_success = True
                    await self.build_twilight_council()
            elif self.already_pending_upgrade(PROTOSSGROUNDWEAPONSLEVEL3) == 0 and self.already_pending_upgrade(PROTOSSGROUNDWEAPONSLEVEL2) == 1:
                if self.units(TWILIGHTCOUNCIL).ready:
                    if self.can_afford(PROTOSSGROUNDWEAPONSLEVEL3):                
                        self.choice_success = True
                        await self.do(forge.first.research(PROTOSSGROUNDWEAPONSLEVEL3))
        else:
            self.choice_success = True
            await self.build_forge()

    # Upgrades armor for all ground units, from level 1 to 3, builds the required structures if they do not exist
    async def upgrade_armor(self):        
        if self.units(FORGE).ready.exists:
            forge = self.units(FORGE).ready.noqueue
            if forge.exists and self.already_pending_upgrade(PROTOSSGROUNDARMORSLEVEL1) == 0:
                if self.can_afford(PROTOSSGROUNDARMORSLEVEL1):
                    self.choice_success = True
                    await self.do(forge.first.research(PROTOSSGROUNDARMORSLEVEL1))
            elif self.already_pending_upgrade(PROTOSSGROUNDARMORSLEVEL2) == 0 and self.already_pending_upgrade(PROTOSSGROUNDARMORSLEVEL1) == 1:
                if self.units(TWILIGHTCOUNCIL).ready:
                    if self.can_afford(PROTOSSGROUNDARMORSLEVEL2):      
                        await self.do(forge.first.research(PROTOSSGROUNDARMORSLEVEL2))
                else:
                    self.choice_success = True
                    await self.build_twilight_council()
            elif self.already_pending_upgrade(PROTOSSGROUNDARMORSLEVEL3) == 0 and self.already_pending_upgrade(PROTOSSGROUNDARMORSLEVEL2) == 1:
                if self.units(TWILIGHTCOUNCIL).ready: 
                    if self.can_afford(PROTOSSGROUNDARMORSLEVEL3):       
                        self.choice_success = True
                        await self.do(forge.first.research(PROTOSSGROUNDARMORSLEVEL3))
        else:
            self.choice_success = True
            await self.build_forge()
        
    # Upgrades shields for all units and structures, from level 1 to 3, builds the required structures if they do not exist
    async def upgrade_shields(self):
        if self.units(FORGE).ready.exists:
            forge = self.units(FORGE).ready.noqueue
            if forge.exists and self.already_pending_upgrade(PROTOSSSHIELDSLEVEL1) == 0:
                if self.can_afford(PROTOSSSHIELDSLEVEL1):
                    self.choice_success = True
                    await self.do(forge.first.research(PROTOSSSHIELDSLEVEL1))
            elif self.already_pending_upgrade(PROTOSSSHIELDSLEVEL2) == 0 and self.already_pending_upgrade(PROTOSSSHIELDSLEVEL1) == 1:
                if self.units(TWILIGHTCOUNCIL).ready:
                    if self.can_afford(PROTOSSSHIELDSLEVEL2):                    
                        await self.do(forge.first.research(PROTOSSSHIELDSLEVEL2))
                else:
                    self.choice_success = True
                    await self.build_twilight_council()
            elif self.already_pending_upgrade(PROTOSSSHIELDSLEVEL3) == 0 and self.already_pending_upgrade(PROTOSSSHIELDSLEVEL2) == 1:
                if self.units(TWILIGHTCOUNCIL).ready:
                    if self.can_afford(PROTOSSSHIELDSLEVEL3):                
                        self.choice_success = True
                        await self.do(forge.first.research(PROTOSSSHIELDSLEVEL3))
        else:
            self.choice_success = True
            await self.build_forge()

    # Upgrades attack for air units, does nothing if the structure required does not exist (will create that later anyway)
    async def upgrade_air_attack(self):
        if self.units(CYBERNETICSCORE).ready.exists:
            cybernetics_core = self.units(CYBERNETICSCORE).ready.noqueue
            if cybernetics_core.exists and self.already_pending_upgrade(PROTOSSAIRWEAPONSLEVEL1) == 0:
                if self.can_afford(PROTOSSAIRWEAPONSLEVEL1):
                    self.choice_success = True
                    await self.do(cybernetics_core.first.research(PROTOSSAIRWEAPONSLEVEL1))
            elif self.already_pending_upgrade(PROTOSSAIRWEAPONSLEVEL2) == 0 and self.already_pending_upgrade(PROTOSSAIRWEAPONSLEVEL1) == 1:
                if self.units(FLEETBEACON).ready:
                    if self.can_afford(PROTOSSAIRWEAPONSLEVEL2):                    
                        self.choice_success = True
                        await self.do(cybernetics_core.first.research(PROTOSSAIRWEAPONSLEVEL2))
            elif self.already_pending_upgrade(PROTOSSAIRWEAPONSLEVEL3) == 0 and self.already_pending_upgrade(PROTOSSAIRWEAPONSLEVEL2) == 1:
                if self.units(FLEETBEACON).ready:
                    if self.can_afford(PROTOSSAIRWEAPONSLEVEL3):                
                        self.choice_success = True
                        await self.do(cybernetics_core.first.research(PROTOSSAIRWEAPONSLEVEL3))

    # Upgrades armor for air units, does nothing if the structure required does not exist (will create that later anyway)
    async def upgrade_air_armor(self):        
        if self.units(CYBERNETICSCORE).ready.exists:
            cybernetics_core = self.units(CYBERNETICSCORE).ready.noqueue
            if cybernetics_core.exists and self.already_pending_upgrade(PROTOSSAIRARMORSLEVEL1) == 0:
                if self.can_afford(PROTOSSAIRARMORSLEVEL1):
                    self.choice_success = True
                    await self.do(cybernetics_core.first.research(PROTOSSAIRARMORSLEVEL1))
            elif self.already_pending_upgrade(PROTOSSAIRARMORSLEVEL2) == 0 and self.already_pending_upgrade(PROTOSSAIRARMORSLEVEL1) == 1:
                if self.units(FLEETBEACON).ready:
                    if self.can_afford(PROTOSSAIRARMORSLEVEL2):                    
                        self.choice_success = True
                        await self.do(cybernetics_core.first.research(PROTOSSAIRARMORSLEVEL2))
            elif self.already_pending_upgrade(PROTOSSAIRARMORSLEVEL3) == 0 and self.already_pending_upgrade(PROTOSSAIRARMORSLEVEL2) == 1:
                if self.units(FLEETBEACON).ready:
                    if self.can_afford(PROTOSSAIRARMORSLEVEL3):                
                        self.choice_success = True
                        await self.do(cybernetics_core.first.research(PROTOSSAIRARMORSLEVEL3))

    # 20 choises the bot can pick from, the model uses this as well as the random choise
    async def do_something(self):
        the_choices = {0: "build_scout",
                       1: "build_zealot",
                       2: "build_gateway",
                       3: "build_voidray",
                       4: "build_stalker",
                       5: "build_worker",
                       6: "build_assimilator",
                       7: "build_airforce",
                       8: "build_pylon",
                       9: "defend_nexus",
                       10: "attack",
                       11: "expand",
                       12: "do_nothing",
                       13: "build_colossus",
                       14: "build_carrier",
                       15: "upgrade_attack",
                       16: "upgrade_armor",
                       17: "upgrade_shields",
                       18: "upgrade_air_attack",
                       19: "upgrade_air_armor"
                      }

        if self.time > self.do_something_after:
            # if the model is being used we can alter the weight of each decision, this was made influence the weight of choises, but this
            # ... is typically not a good idea, that is way we have it at 1 now
            if self.use_model:
                worker_weight = 1
                zealot_weight = 1
                voidray_weight = 1
                stalker_weight = 1
                pylon_weight = 1
                stargate_weight = 1
                gateway_weight = 1
                assimilator_weight = 1

                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 1])])
                weights = [1, zealot_weight, gateway_weight, voidray_weight, stalker_weight, worker_weight, assimilator_weight, stargate_weight, pylon_weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                weighted_prediction = prediction[0]*weights
                choice = np.argmax(weighted_prediction)
                print('Choice:',the_choices[choice])
            else:
                # if the random factor is being used we can alter the weight of each decision, this was made influence the weight of choises, 
                # ... but this is typically not a good idea, that is way we have it at 1 now, except the void ray
                worker_weight = 1 #8
                zealot_weight = 1 #3
                voidray_weight = 2 #20
                stalker_weight = 1 #8
                pylon_weight = 1 #5
                stargate_weight = 1 #5
                gateway_weight = 1 #3

                choice_weights = 1*[0]+zealot_weight*[1]+gateway_weight*[2]+voidray_weight*[3]+stalker_weight*[4]+worker_weight*[5]+1*[6]+stargate_weight*[7]+pylon_weight*[8]+1*[9]+1*[10]+1*[11]+1*[12]+1*[13]+1*[14]+1*[15]+1*[16]+1*[17]+1*[18]+1*[19]
                choice = random.choice(choice_weights)

            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e))

             ###############################################################################################################################
            # We tried to make choises with the if statement below, that is we only add a choice if it succeeds. We did this to             #
            # remove noise in our data. The problem here is that we weren't able to get our training to run if we did this, even            #
            # though the training data seems to look the same whether we did this or not. The reason for the noise is because choises       #
            # where saved even though they where not completed successfully which made us store a choice when nothing happened, we wanted   #
            # to optimize this so that choices made that were a success where stored.                                                       #
             ###############################################################################################################################

            # if self.choice_success:
            #     y = np.zeros(20)
            #     y[choice] = 1
            #     self.train_data.append([y, self.flipped])
            #     self.choice_success = False

            # creates an array of size 20 (number of choises) and makes them all 0
            y = np.zeros(20)
            y[choice] = 1   # The choice made is stored as 1

            # we append the choice made along with the image that was in the cv2 window at that given moment
            self.train_data.append([y, self.flipped])

# Here we run the game, load up the map, our bot and a computer opponent.
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ADBot()),
    Computer(Race.Zerg, Difficulty.Easy)
], realtime=False)