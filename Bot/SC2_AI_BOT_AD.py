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

os.environ["SC2PATH"] = 'D:/StarCraft II/'

HEADLESS = False

# Daniel PC command
# D:/Python/Python37-32/python.exe c:/Users/danie/Desktop/StarCraft2_AI_BOT/Bot/SC2_AI_BOT_AD.py

def get_session(gpu_fraction=0.3):
    """Assume that you have 6GB of GPU memory and want to allocate ~2GB"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())


class ADBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.use_model = use_model
        self.title = title
        self.military_units = [ZEALOT, VOIDRAY, STALKER, CARRIER, COLOSSUS]
        self.choice_success = False

        ###############################
        # DICT {UNIT_ID:LOCATION}
        # every iteration, make sure that unit id still exists!
        self.scouts_and_spots = {}

        # ADDED THE CHOICES #
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
                        11: self.expand,  # might just be self.expand_now() lol
                        12: self.do_nothing,
                        13: self.build_colossus,
                        14: self.build_carrier,
                        15: self.upgrade_attack,
                        16: self.upgrade_armor,
                        17: self.upgrade_shields,
                        18: self.upgrade_air_attack,
                        19: self.upgrade_air_armor
                        }

        self.train_data = []
        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("AD-100-epochs-0.001-LR-STAGE1.model")


    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

        if self.use_model:
            with open("gameout-model-vs-easy.txt","a") as f:
                f.write("Model {} - {}\n".format(game_result, int(time.time())))

    async def on_step(self, iteration):

        # print('Time:',self.state.game_loop)
        # self.time = (self.state.game_loop/22.4) / 60
        # print('Time:',self.time)

        if iteration % 5 == 0:
            await self.distribute_workers()
        await self.scout()
        await self.intel()
        await self.do_something()

    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        #  FIXED THIS
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


    async def scout(self):
        '''
        ['__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_game_data', '_proto', '_type_data', 'add_on_tag', 'alliance', 'assigned_harvesters', 'attack', 'build', 'build_progress', 'cloak', 'detect_range', 'distance_to', 'energy', 'facing', 'gather', 'has_add_on', 'has_buff', 'health', 'health_max', 'hold_position', 'ideal_harvesters', 'is_blip', 'is_burrowed', 'is_enemy', 'is_flying', 'is_idle', 'is_mine', 'is_mineral_field', 'is_powered', 'is_ready', 'is_selected', 'is_snapshot', 'is_structure', 'is_vespene_geyser', 'is_visible', 'mineral_contents', 'move', 'name', 'noqueue', 'orders', 'owner_id', 'position', 'radar_range', 'radius', 'return_resource', 'shield', 'shield_max', 'stop', 'tag', 'train', 'type_id', 'vespene_contents', 'warp_in']
        '''
        self.expand_dis_dir = {}

        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            #print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

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

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
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

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))


    async def intel(self):
        '''
        just simply iterate units.

        outline fighters in white possibly?

        draw pending units with more alpha

        '''

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)


        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))


        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:
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


        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        #print(self.flipped)

        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)


        if not HEADLESS:
            if self.use_model:
                cv2.imshow(str(self.title), resized)
                cv2.waitKey(1)
            else:
                cv2.imshow(str(self.title), resized)
                cv2.waitKey(1)

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

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



    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE):
                self.choice_success = True
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_zealot(self):
        #if len(self.units(ZEALOT)) < (8 - self.time): # how we can phase out zealots over time?
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                self.choice_success = True
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_gateway(self):
        if len(self.units(GATEWAY)) < len(self.units(NEXUS).ready) * 2:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                self.choice_success = True
                await self.build(GATEWAY, target)

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready.noqueue
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                self.choice_success = True
                await self.do(random.choice(stargates).train(VOIDRAY))
        #####
        else:
            await self.build_airforce()

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
    
    async def build_carrier(self):
        stargates = self.units(STARGATE).ready
        fleetbeacons = self.units(FLEETBEACON)
        if stargates.exists and fleetbeacons.exists:
            if self.can_afford(CARRIER):
                self.choice_success = True
                await self.do(random.choice(stargates).train(CARRIER))
            else:
                await self.build_airforce()

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

    async def build_robotics_bay(self):
        robotics_facility = self.units(ROBOTICSFACILITY)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if robotics_facility.ready.exists:
                if not self.units(ROBOTICSBAY).exists:
                    if self.can_afford(ROBOTICSBAY):
                        target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                        await self.build(ROBOTICSBAY, target)

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

    async def build_forge(self):
        if self.units(NEXUS).ready.exists and self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.can_afford(FORGE) and not self.already_pending(FORGE) and len(self.units(NEXUS)) > len(self.units(FORGE)) and len(self.known_enemy_units(FORGE)) < 2:
                target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                await self.build(FORGE, target)
                
    async def build_twilight_council(self):
        if self.units(CYBERNETICSCORE).ready.exists and self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            twilightcouncil = self.units(TWILIGHTCOUNCIL)
            if not twilightcouncil.exists:
                if self.can_afford(TWILIGHTCOUNCIL) and not self.already_pending(TWILIGHTCOUNCIL):
                    target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                    await self.build(TWILIGHTCOUNCIL, target)

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

            ########################################
            if not cybernetics_cores.exists:
                if self.units(GATEWAY).ready.exists:
                    if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                        target = await self.find_placement(PYLON, near=pylon.position.towards(pylon.position.random_on_distance(pylon.position), 5))
                        self.choice_success = True
                        await self.build(CYBERNETICSCORE, target)

    async def build_pylon(self):
        # changed supply left, and added a random position with more distance to either pylons or nexuses
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

    async def expand(self):
        try:
            if self.can_afford(NEXUS) and len(self.units(NEXUS)):
                self.choice_success = True
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.choice_success = True
        self.do_something_after = self.time + wait

    async def defend_nexus(self):
        if len(self.known_enemy_units) > 0:
            nexus = random.choice(self.units(NEXUS))
            target = self.known_enemy_units.closest_to(nexus)
            if nexus.distance_to(target) < 25:
                for unit_type in self.military_units:
                    for u in self.units(unit_type).idle: 
                        self.choice_success = True
                        await self.do(u.attack(target))

    async def attack(self):
        for unit_type in self.military_units:
            if len(self.units(unit_type)) > 4:
                if len(self.known_enemy_units) > 0:
                    self.choice_success = True
                    await self.attack_known_enemy_unit()

    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            for unit_type in self.military_units:
                for u in self.units(unit_type).idle:
                    target = random.choice(self.known_enemy_units).position
                    await self.do(u.scan_move(target))

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


            y = np.zeros(20)
            y[choice] = 1
            self.train_data.append([y, self.flipped])
            self.choice_success = False


run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ADBot()),
    Computer(Race.Zerg, Difficulty.Easy)
], realtime=False)
