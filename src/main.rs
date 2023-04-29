mod brain;
mod creature;
mod utils;
mod world;

use crate::brain::NeuralBrain;
use crate::creature::*;
use crate::world::*;
use bevy::math::vec3;
use bevy::prelude::*;
use bevy_prototype_debug_lines::*;

const TIME_STEP: f32 = 1.0 / 60.0;
const BACKGROUND_COLOR: Color = Color::rgb(0.0, 0.0, 0.0);

fn main() {
    let handle_food = (
        decay_food,
        populate_food.after(decay_food),
        food_despawn.after(decay_food),
    );
    let handle_creature_decay = (decay_creatures, creature_despawn.after(decay_creatures));
    let perception_and_actions = (
        populate_grid,
        self_perception,
        vision_perception.after(populate_grid),
        think_of_actions
            .after(vision_perception)
            .after(self_perception),
        move_from_actions.after(think_of_actions),
        find_closest_food.after(think_of_actions),
        find_closest_creature.after(think_of_actions),
    );

    App::new()
        // Add window etc..
        .add_plugins(DefaultPlugins)
        // Debug lines
        .add_plugin(DebugLinesPlugin::default())
        // Clear screen using ClearColor
        .insert_resource(ClearColor(BACKGROUND_COLOR))
        .insert_resource(FixedTime::new_from_secs(TIME_STEP))
        .insert_resource(MaxFood {
            total_energy: 600_000,
            min_food_grow: 600,
            max_food_grow: 6000,
            food_std: 600.0,
        })
        .insert_resource(CreaturePreferences::default())
        .insert_resource(FoodCount::default())
        .insert_resource(CollisionGrid::new(1000.0, 100))
        .insert_resource(FixedTime::new_from_secs(TIME_STEP))
        .add_startup_system(setup)
        .add_systems(handle_food.in_schedule(CoreSchedule::FixedUpdate))
        .add_systems(handle_creature_decay.in_schedule(CoreSchedule::FixedUpdate))
        .add_systems(perception_and_actions.in_schedule(CoreSchedule::FixedUpdate))
        // Debugging
        .add_system(draw_vision_lines)
        .add_system(move_camera)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    creature_prefs: Res<CreaturePreferences>,
) {
    let num_vision_inputs =
        VisionPerception::num_channels_per_angle() * creature_prefs.vision_slices;
    let num_self_inputs =
        SelfPerception::num_non_memory_perceptions() + creature_prefs.num_memories;
    let num_actions = Actions::num_non_memory_actions() + creature_prefs.num_memories;

    commands.spawn(Camera2dBundle::default());
    commands.spawn(Creature {
        brain: NeuralBrain::new_random(num_vision_inputs + num_self_inputs, num_actions, &[16]),
        vperception: Default::default(),
        sperception: Default::default(),
        actions: Default::default(),
        view_color: ViewColor {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        },
        sprite: SpriteBundle {
            texture: asset_server.load("imgs/animal.png"),
            sprite: Sprite {
                color: Color::rgb(1.0, 0.0, 0.0),
                ..default()
            },
            transform: Transform {
                translation: Vec3::from([0., 0., 0.]),
                scale: Vec3::from([0.25, 0.25, 1.0]),
                rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            },
            ..default()
        },
        dets: CreatureDetails {
            energy: 600000.0,
            age: 0,
        },
        target_food: Default::default(),
        target_creature: Default::default(),
    });
}

fn draw_vision_lines(
    query: Query<(&VisionPerception, &Transform, &TargetFood, &TargetCreature)>,
    foods: Query<&Transform, With<FoodEnergy>>,
    creatures: Query<&Transform, With<CreatureDetails>>,
    mut lines: ResMut<DebugLines>,
    creature_prefs: Res<CreaturePreferences>,
) {
    let mut line3 = |p1: Vec3, p2: Vec3, c: Color| {
        lines.line_colored(vec3(p1.x, p1.y, 0.0), vec3(p2.x, p2.y, 0.0), 0.0, c)
    };

    let dangle = (2. * creature_prefs.vision_range) / (creature_prefs.vision_slices as f32);
    query
        .iter()
        .take(1)
        .for_each(|(perc, t, target_food, target_creature)| {
            let invt = t.rotation;
            (0..perc.n())
                .filter(|i| perc.d()[*i].is_finite())
                .for_each(|i| {
                    let color = Color::rgb(perc.r()[i], perc.g()[i], perc.b()[i]);
                    let v = vec3(perc.d()[i], 0.0, 0.0);
                    let r = Quat::from_rotation_z(
                        dangle * (i as f32) - creature_prefs.vision_range + dangle / 2.0,
                    );
                    let v = r.mul_vec3(v);
                    let v = invt.mul_vec3(v);
                    line3(t.translation, t.translation + v, color);
                });

            let v = vec3(creature_prefs.max_view_dist, 0.0, 0.0);
            let v = invt.mul_vec3(v);
            let r = Quat::from_rotation_z(-creature_prefs.vision_range);
            let va = r.mul_vec3(v);
            let vb = r.inverse().mul_vec3(v);
            line3(t.translation, t.translation + va, Color::WHITE);
            line3(t.translation, t.translation + vb, Color::WHITE);

            let v = vec3(creature_prefs.mouth_radius, 0.0, 0.0);
            let v = invt.mul_vec3(v);
            let r = Quat::from_rotation_z(-creature_prefs.vision_range);
            let va = r.mul_vec3(v);
            let vb = r.inverse().mul_vec3(v);
            line3(t.translation, t.translation + va, Color::RED);
            line3(t.translation + va, t.translation + vb, Color::RED);
            line3(t.translation, t.translation + vb, Color::RED);

            let food = target_food.target.and_then(|e| foods.get(e).ok());
            if let Some(food) = food {
                let tl = food.translation + Vec3::new(-10., 10., 0.);
                let tr = food.translation + Vec3::new(10., 10., 0.);
                let br = food.translation + Vec3::new(10., -10., 0.);
                let bl = food.translation + Vec3::new(-10., -10., 0.);
                line3(tl, tr, Color::GREEN);
                line3(tr, br, Color::GREEN);
                line3(br, bl, Color::GREEN);
                line3(bl, tl, Color::GREEN);
            }
            let creature = target_creature.target.and_then(|e| creatures.get(e).ok());
            if let Some(creature) = creature {
                let tl = creature.translation + Vec3::new(-10., 10., 0.);
                let tr = creature.translation + Vec3::new(10., 10., 0.);
                let br = creature.translation + Vec3::new(10., -10., 0.);
                let bl = creature.translation + Vec3::new(-10., -10., 0.);
                line3(tl, tr, Color::RED);
                line3(tr, br, Color::RED);
                line3(br, bl, Color::RED);
                line3(bl, tl, Color::RED);
            }
        });
}

fn move_camera(mut camera_query: Query<&mut Transform, With<Camera>>, keys: Res<Input<KeyCode>>) {
    for mut camera_transform in camera_query.iter_mut() {
        if keys.pressed(KeyCode::W) {
            camera_transform.translation.y += 10.0 * camera_transform.scale.y;
        }
        if keys.pressed(KeyCode::S) {
            camera_transform.translation.y -= 10.0 * camera_transform.scale.y;
        }
        if keys.pressed(KeyCode::A) {
            camera_transform.translation.x -= 10.0 * camera_transform.scale.x;
        }
        if keys.pressed(KeyCode::D) {
            camera_transform.translation.x += 10.0 * camera_transform.scale.x;
        }
        if keys.pressed(KeyCode::Q) {
            camera_transform.scale.x *= 1.01;
            camera_transform.scale.y *= 1.01;
        }
        if keys.pressed(KeyCode::E) {
            camera_transform.scale.x *= 1.0 / 1.01;
            camera_transform.scale.y *= 1.0 / 1.01;
        }
    }
}
