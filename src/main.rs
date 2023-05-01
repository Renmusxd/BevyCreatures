mod brain;
mod creature;
mod utils;
mod world;

use crate::creature::*;
use crate::utils::*;
use crate::world::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::math::vec3;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_prototype_debug_lines::*;
use num::traits::Pow;
use std::cmp::Ordering::Equal;

fn main() {
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
        creatures_eat.after(find_closest_food),
        creatures_bite.after(find_closest_creature),
        eat_drained.after(creatures_bite),
        creatures_split.after(creatures_eat).after(eat_drained),
    );
    let handle_food = (
        decay_food,
        populate_food.after(decay_food),
        food_despawn.after(decay_food),
    );
    let handle_creature_decay = (
        decay_creatures,
        creature_despawn.after(decay_creatures),
        repopulate_creatures.after(decay_creatures),
    );

    App::new()
        // Add window etc..
        .add_plugins(DefaultPlugins)
        // Debug lines
        .add_plugin(DebugLinesPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        // Clear screen using ClearColor
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(FixedTime::new_from_secs(1. / 360.))
        .insert_resource(MaxFood {
            total_energy: 1_000_000,
            min_food_grow: 500,
            max_food_grow: 5000,
            food_std: 1000.0,
        })
        .insert_resource(CreatureCount {
            count: 0,
            min_count: 100,
        })
        .insert_resource(Game::default())
        .insert_resource(CreaturePreferences::default())
        .insert_resource(FoodCount::default())
        .insert_resource(CollisionGrid::new(2000.0, 100))
        .add_startup_system(setup)
        .add_systems(perception_and_actions.in_schedule(CoreSchedule::FixedUpdate))
        .add_systems(
            handle_food
                .after(creatures_eat)
                .in_schedule(CoreSchedule::FixedUpdate),
        )
        .add_systems(
            handle_creature_decay
                .after(creatures_split)
                .in_schedule(CoreSchedule::FixedUpdate),
        )
        // Debugging
        .add_system(draw_vision_lines)
        .add_system(choose_creature)
        .add_system(move_camera)
        .run();
}

#[derive(Component)]
struct MainCamera;

fn setup(mut commands: Commands) {
    commands.spawn((Camera2dBundle::default(), MainCamera));
}

fn draw_vision_lines(
    query: Query<(&VisionPerception, &Transform, &TargetFood, &CreatureTarget)>,
    foods: Query<&Transform, With<FoodEnergy>>,
    creatures: Query<&Transform, With<CreatureDetails>>,
    mut lines: ResMut<DebugLines>,
    creature_prefs: Res<CreaturePreferences>,
    chosen_creature: Res<Game>,
) {
    let mut line3 = |p1: Vec3, p2: Vec3, c: Color| {
        lines.line_colored(vec3(p1.x, p1.y, 0.0), vec3(p2.x, p2.y, 0.0), 0.0, c)
    };

    let dangle = (2. * creature_prefs.vision_range) / (creature_prefs.vision_slices as f32);
    let creature = chosen_creature
        .chosen_creature
        .and_then(|e| query.get(e).ok());

    if let Some((perc, t, target_food, target_creature)) = creature {
        let invt = t.rotation;
        (0..perc.n())
            .filter(|i| perc.d()[*i].is_finite())
            .for_each(|i| {
                let distance = (1. - perc.d()[i]) * creature_prefs.max_view_dist;
                let color = Color::rgb(perc.r()[i], perc.g()[i], perc.b()[i]);
                let v = vec3(distance, 0.0, 0.0);
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
    }
}

#[derive(Resource, Default)]
struct Game {
    chosen_creature: Option<Entity>,
}

fn choose_creature(
    primary_query: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mouse_click: Res<Input<MouseButton>>,
    grids: Res<CollisionGrid>,
    mut chosen_creature: ResMut<Game>,
) {
    if mouse_click.just_pressed(MouseButton::Left) {
        let (camera, camera_transform) = camera_q.single();
        let window = primary_query.single();
        if let Some(cursor) = window
            .cursor_position()
            .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
            .map(|ray| ray.origin.truncate())
        {
            chosen_creature.chosen_creature = grids
                .creaturegrid
                .within_dist_of(cursor.x, cursor.y, 100.)
                .map(|(tx, ty, e)| -> (f32, Entity) {
                    let d2 = (*tx - cursor.x).pow(2) + (*ty - cursor.y).pow(2);
                    (d2, *e)
                })
                .min_by(|(d2a, _), (d2b, _)| d2a.partial_cmp(d2b).unwrap_or(Equal))
                .map(|(_, e)| e);
        }
    }
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
