use crate::brain::NeuralBrain;
use crate::utils::Grid;
use crate::world::{FoodEnergy, ViewColor};
use bevy::math::{vec2, vec3, Vec3Swizzles};
use bevy::prelude::*;
use num::pow::Pow;
use num::Float;
use std::cmp::Ordering::Equal;

#[derive(Bundle)]
pub struct Creature {
    pub(crate) brain: NeuralBrain,
    pub(crate) vperception: VisionPerception,
    pub(crate) sperception: SelfPerception,
    pub(crate) actions: Actions,
    pub(crate) sprite: SpriteBundle,
    pub(crate) view_color: ViewColor,
    pub(crate) dets: CreatureDetails,
    pub(crate) target_food: TargetFood,
    pub(crate) target_creature: TargetCreature,
}

#[derive(Component)]
pub struct CreatureDetails {
    pub energy: f32,
    pub age: usize,
}

#[derive(Component, Default)]
pub struct TargetFood {
    pub(crate) target: Option<Entity>,
}
#[derive(Component, Default)]
pub struct TargetCreature {
    pub(crate) target: Option<Entity>,
}

#[derive(Component, Default)]
pub struct VisionPerception {
    pub dat: Vec<f32>,
    n: usize,
}

impl VisionPerception {
    fn init(&mut self, n: usize) {
        self.clear();
        self.n = n;
        self.dat
            .resize((Self::num_channels_per_angle() - 1) * n, 0.0);
        self.dat
            .resize(Self::num_channels_per_angle() * n, f32::infinity());
    }
    pub fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn r(&self) -> &[f32] {
        &self.dat[0..self.n]
    }
    fn r_mut(&mut self) -> &mut [f32] {
        &mut self.dat[0..self.n]
    }
    pub(crate) fn g(&self) -> &[f32] {
        &self.dat[self.n..2 * self.n]
    }
    fn g_mut(&mut self) -> &mut [f32] {
        &mut self.dat[self.n..2 * self.n]
    }
    pub(crate) fn b(&self) -> &[f32] {
        &self.dat[2 * self.n..3 * self.n]
    }
    fn b_mut(&mut self) -> &mut [f32] {
        &mut self.dat[2 * self.n..3 * self.n]
    }
    pub(crate) fn d(&self) -> &[f32] {
        &self.dat[3 * self.n..]
    }
    fn d_mut(&mut self) -> &mut [f32] {
        &mut self.dat[3 * self.n..]
    }

    fn clear(&mut self) {
        self.dat.clear()
    }

    pub fn num_channels_per_angle() -> usize {
        4
    }
}

#[derive(Component, Default)]
pub struct SelfPerception {
    energy: f32,
    age: f32,
    being_attacked: bool,
    memory: Vec<f32>,
}

impl SelfPerception {
    pub fn num_non_memory_perceptions() -> usize {
        3
    }
    fn get_dat(&self) -> ([f32; 3], &[f32]) {
        (
            [
                self.energy,
                self.age,
                if self.being_attacked { 1.0 } else { 0.0 },
            ],
            &self.memory,
        )
    }
}

#[derive(Component, Default, Debug)]
pub struct Actions {
    left_motor: f32,
    right_motor: f32,
    bite: bool,
    feed: bool,
    split: bool,
    memory: Vec<f32>,
}

impl Actions {
    pub fn num_non_memory_actions() -> usize {
        5
    }
}

#[derive(Resource)]
pub struct CreaturePreferences {
    pub(crate) max_view_dist: f32,
    pub(crate) vision_range: f32,    // angle left or right of center
    pub(crate) vision_slices: usize, // number of chunks for vision
    pub(crate) energy_scale: f32,
    pub(crate) max_age: usize,
    pub(crate) walk_speed: f32,
    pub(crate) turn_speed: f32,
    pub(crate) num_memories: usize,
    pub(crate) mouth_radius: f32,
    pub(crate) energy_costs: EnergyCosts,
}
impl Default for CreaturePreferences {
    fn default() -> Self {
        CreaturePreferences {
            max_view_dist: 300.0,
            vision_range: std::f32::consts::PI / 8.0,
            vision_slices: 7,
            energy_scale: 10_000.,
            max_age: 1000_000,
            walk_speed: 0.1,
            turn_speed: 0.01,
            num_memories: 5,
            mouth_radius: 20.,
            energy_costs: Default::default(),
        }
    }
}

pub struct EnergyCosts {
    base_cost: f32,
    motor_cost_scale: f32,
    bite_cost: f32,
    feed_cost: f32,
}
impl Default for EnergyCosts {
    fn default() -> Self {
        EnergyCosts {
            base_cost: 1.0,
            motor_cost_scale: 10.0,
            bite_cost: 10.0,
            feed_cost: 10.0,
        }
    }
}

// Remove energy based on actions.
pub fn decay_creatures(
    mut query: Query<(&mut CreatureDetails, &Actions)>,
    creature_preferences: Res<CreaturePreferences>,
) {
    let costs = &creature_preferences.energy_costs;
    query.par_iter_mut().for_each_mut(|(mut cd, acts)| {
        let mut acc = costs.base_cost;
        acc += costs.motor_cost_scale * (acts.left_motor.pow(2) + acts.right_motor.pow(2));
        if acts.bite {
            acc += costs.bite_cost;
        }
        if acts.feed {
            acc += costs.feed_cost;
        }
        cd.energy -= acc;
    })
}

// Remove creatures with zero energy.
pub fn creature_despawn(
    query: Query<(Entity, &CreatureDetails)>,
    mut commands: Commands,
    creature_preferences: Res<CreaturePreferences>,
) {
    for (entity, cd) in query.iter() {
        if cd.energy <= 0. || cd.age >= creature_preferences.max_age {
            commands.entity(entity).despawn();
        }
    }
}

#[derive(Resource)]
pub struct CollisionGrid {
    foodgrid: Grid<Entity>,
    creaturegrid: Grid<Entity>,
}

impl CollisionGrid {
    pub(crate) fn new(r: f32, n: usize) -> Self {
        Self {
            foodgrid: Grid::new(-r, r, -r, r, n, n),
            creaturegrid: Grid::new(-r, r, -r, r, n, n),
        }
    }
}

pub fn populate_grid(
    creatures: Query<(Entity, &Transform), With<CreatureDetails>>,
    foods: Query<(Entity, &Transform), With<FoodEnergy>>,
    mut grids: ResMut<CollisionGrid>,
) {
    grids.creaturegrid.clear();
    creatures.iter().for_each(|(e, t)| {
        grids
            .creaturegrid
            .insert(t.translation.x, t.translation.y, e);
    });
    grids.foodgrid.clear();
    foods.iter().for_each(|(e, t)| {
        grids.foodgrid.insert(t.translation.x, t.translation.y, e);
    });
}

// Look outward
pub fn vision_perception(
    mut perceivee: Query<(Entity, &mut VisionPerception, &Transform)>,
    can_see: Query<&ViewColor>,
    grids: Res<CollisionGrid>,
    creature_prefs: Res<CreaturePreferences>,
) {
    let dangle = (2. * creature_prefs.vision_range) / (creature_prefs.vision_slices as f32);
    perceivee
        .par_iter_mut()
        .for_each_mut(|(entity, mut perc, t)| {
            perc.clear();
            perc.init(creature_prefs.vision_slices);

            let v = t.translation.xy();
            let theta = t.rotation;
            let creatures =
                grids
                    .creaturegrid
                    .within_dist_of(v.x, v.y, creature_prefs.max_view_dist);
            let foods = grids
                .foodgrid
                .within_dist_of(v.x, v.y, creature_prefs.max_view_dist);
            creatures
                .chain(foods)
                .filter(|(_, _, e)| entity.ne(e))
                .map(|(tx, ty, e)| {
                    let color = can_see.get(*e).expect("Entity not found");
                    (tx, ty, color)
                })
                .for_each(|(tx, ty, c)| {
                    let tv = vec2(*tx, *ty);
                    let dtheta = relative_angle(v, tv, theta);
                    let above_min = dtheta >= -creature_prefs.vision_range;
                    let below_max = dtheta < creature_prefs.vision_range;
                    if above_min && below_max {
                        let zeroed_dtheta = dtheta + creature_prefs.vision_range;
                        let count_zeroed = (zeroed_dtheta / dangle).floor() as usize;
                        let d = (tv - v).length();
                        if d < perc.d()[count_zeroed] {
                            perc.r_mut()[count_zeroed] = c.r;
                            perc.g_mut()[count_zeroed] = c.g;
                            perc.b_mut()[count_zeroed] = c.b;
                            perc.d_mut()[count_zeroed] = d;
                        }
                    }
                });
        });
}

// Look inward
pub fn self_perception(
    mut perceivee: Query<(&mut SelfPerception, &CreatureDetails, &Actions)>,
    creature_prefs: Res<CreaturePreferences>,
) {
    perceivee
        .par_iter_mut()
        .for_each_mut(|(mut perc, dets, acts)| {
            perc.energy = (dets.energy as f32) / (creature_prefs.energy_scale as f32);
            perc.age = (dets.age as f32) / (creature_prefs.max_age as f32);
            perc.memory.clear();
            perc.memory.extend_from_slice(&acts.memory);
            perc.memory.resize(creature_prefs.num_memories, 0.);
        });
}

pub fn think_of_actions(
    mut thinkers: Query<(
        &VisionPerception,
        &SelfPerception,
        &NeuralBrain,
        &mut Actions,
    )>,
    creature_prefs: Res<CreaturePreferences>,
) {
    thinkers
        .par_iter_mut()
        .for_each_mut(|(vperc, sperc, brain, mut acts)| {
            let vision_inputs = &vperc.dat;
            let (self_inputs, memories) = sperc.get_dat();
            let iter = vision_inputs
                .iter()
                .chain(self_inputs.iter())
                .chain(memories.iter())
                .copied()
                .map(|v| if !v.is_finite() { 0. } else { v });
            // Reuse memory array for storing output from brain.
            acts.memory.clear();
            acts.memory.resize(
                Actions::num_non_memory_actions() + creature_prefs.num_memories,
                0.,
            );
            brain.feed_iter(iter, &mut acts.memory);
            acts.left_motor = acts.memory[creature_prefs.num_memories];
            acts.right_motor = acts.memory[creature_prefs.num_memories + 1];
            acts.bite = acts.memory[creature_prefs.num_memories + 2] > 0.;
            acts.feed = acts.memory[creature_prefs.num_memories + 3] > 0.;
            acts.split = acts.memory[creature_prefs.num_memories + 4] > 0.;
            acts.memory.resize(creature_prefs.num_memories, 0.);
        });
}

pub fn move_from_actions(
    mut actors: Query<(&Actions, &mut Transform)>,
    creature_prefs: Res<CreaturePreferences>,
) {
    actors.par_iter_mut().for_each_mut(|(act, mut t)| {
        let forward = act.right_motor * creature_prefs.walk_speed
            + act.left_motor * creature_prefs.walk_speed;
        let turn = act.right_motor * creature_prefs.turn_speed
            - act.left_motor * creature_prefs.turn_speed;
        let v = t.rotation.mul_vec3(Vec3::X * forward);
        t.translation += v;
        t.rotation *= Quat::from_rotation_z(turn);
    });
}

pub fn creatures_split(actors: Query<&Actions>) {
    todo!()
}

pub fn creatures_bite(actors: Query<&Actions>, grids: Res<CollisionGrid>) {
    todo!()
}

pub fn find_closest_food(
    mut actors: Query<(&Actions, &Transform, &mut TargetFood)>,
    grids: Res<CollisionGrid>,
    creature_prefs: Res<CreaturePreferences>,
) {
    actors
        .par_iter_mut()
        .for_each_mut(|(acts, t, mut target_food)| {
            if acts.feed {
                target_food.target = grids
                    .foodgrid
                    .within_dist_of(
                        t.translation.x,
                        t.translation.y,
                        creature_prefs.mouth_radius,
                    )
                    .filter(|(tx, ty, _)| {
                        let dtheta = relative_angle(t.translation.xy(), vec2(*tx, *ty), t.rotation);
                        let above_min = dtheta >= -creature_prefs.vision_range;
                        let below_max = dtheta < creature_prefs.vision_range;
                        above_min && below_max
                    })
                    .map(|(tx, ty, e)| -> (f32, Entity) {
                        let d2 = (*tx - t.translation.x).pow(2) + (*ty - t.translation.y).pow(2);
                        (d2, *e)
                    })
                    .min_by(|(d2a, _), (d2b, _)| d2a.partial_cmp(d2b).unwrap_or(Equal))
                    .map(|(_, e)| e);
            } else {
                target_food.target = None;
            }
        })
}

pub fn find_closest_creature(
    mut actors: Query<(&Actions, &Transform, &mut TargetCreature)>,
    grids: Res<CollisionGrid>,
    creature_prefs: Res<CreaturePreferences>,
) {
    actors
        .par_iter_mut()
        .for_each_mut(|(acts, t, mut target_creature)| {
            if acts.bite {
                target_creature.target = grids
                    .creaturegrid
                    .within_dist_of(
                        t.translation.x,
                        t.translation.y,
                        creature_prefs.mouth_radius,
                    )
                    .filter(|(tx, ty, _)| {
                        let dtheta = relative_angle(t.translation.xy(), vec2(*tx, *ty), t.rotation);
                        let above_min = dtheta >= -creature_prefs.vision_range;
                        let below_max = dtheta < creature_prefs.vision_range;
                        above_min && below_max
                    })
                    .map(|(tx, ty, e)| -> (f32, Entity) {
                        let d2 = (*tx - t.translation.x).pow(2) + (*ty - t.translation.y).pow(2);
                        (d2, *e)
                    })
                    .min_by(|(d2a, _), (d2b, _)| d2a.partial_cmp(d2b).unwrap_or(Equal))
                    .map(|(_, e)| e);
            } else {
                target_creature.target = None;
            }
        })
}

pub fn creatures_eat(
    mut actors: Query<(&Actions, &Transform, &mut CreatureDetails, &TargetFood)>,
    mut food: Query<&mut FoodEnergy>,
    creature_prefs: Res<CreaturePreferences>,
) {
    actors.for_each_mut(|(acts, t, mut dets, target)| {
        let food = target.target.and_then(|e| food.get(e).ok());
        if let Some(food) = food {
            todo!()
        }
    })
}

#[inline]
fn relative_angle(a: Vec2, b: Vec2, a_angle: Quat) -> f32 {
    let dv = b - a;
    let dv = vec3(dv.x, dv.y, 0.);
    let rdv = a_angle.inverse().mul_vec3(dv);
    let rdv = vec2(rdv.x, rdv.y);
    Vec2::X.angle_between(rdv)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_angles() {
        let v1 = vec2(1.0, 0.0);
        let v2 = vec2(0.5, 0.1);
        let v3 = vec2(0.5, -0.1);

        assert!(v1.angle_between(v2) > 0.);
        assert!(v1.angle_between(v3) < 0.);
    }

    #[test]
    fn test_relative_angle() {
        let v1 = vec2(1.0, 0.0);
        let ta = Quat::from_rotation_z(std::f32::consts::PI);
        let v2 = vec2(0.5, -0.1);
        let v3 = vec2(0.5, 0.1);
        let dangle = relative_angle(v1, v2, ta);
        assert!(dangle > 0.);
        let dangle = relative_angle(v1, v3, ta);
        assert!(dangle < 0.);
    }

    #[test]
    fn test_relative_angle_adv() {
        let v1 = vec2(1.0, 0.0);
        let ta = Quat::from_rotation_z(3. / 2. * std::f32::consts::PI);
        let v2 = vec2(1.1, -1.0);
        let v3 = vec2(-1.1, -1.0);
        let dangle = relative_angle(v1, v2, ta);
        assert!(dangle > 0., "{:?}", dangle);
        let dangle = relative_angle(v1, v3, ta);
        assert!(dangle < 0., "{:?}", dangle);
    }
}
