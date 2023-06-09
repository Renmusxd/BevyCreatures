use bevy::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

#[derive(Default, Resource)]
pub struct FoodCount {
    pub(crate) total_energy: usize,
}
#[derive(Resource)]
pub struct MaxFood {
    pub(crate) total_energy: usize,
    pub(crate) food_grow: usize,
    pub(crate) food_std: f32,
}

#[derive(Component)]
pub struct FoodEnergy {
    pub energy: usize,
}

#[derive(Bundle)]
pub struct Food {
    energy: FoodEnergy,
    view_color: ViewColor,
    #[bundle]
    sprite: SpriteBundle,
}

#[derive(Component)]
pub struct ViewColor {
    pub color: Color,
}

pub fn populate_food(
    mut commands: Commands,
    mut foodcount: ResMut<FoodCount>,
    maxfood: Res<MaxFood>,
    asset_server: Res<AssetServer>,
) {
    if foodcount.total_energy <= maxfood.total_energy - maxfood.food_grow {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, maxfood.food_std).unwrap();
        let x = normal.sample(&mut rng);
        let y = normal.sample(&mut rng);

        let food = Food {
            energy: FoodEnergy {
                energy: maxfood.food_grow,
            },
            view_color: ViewColor {
                color: Color::GREEN,
            },
            sprite: SpriteBundle {
                texture: asset_server.load("imgs/food.png"),
                transform: Transform {
                    translation: Vec3::from([x, y, 0.0]),
                    scale: Vec3::from([0.1, 0.1, 1.0]),
                    ..default()
                },
                ..default()
            },
        };
        commands.spawn(food);
        foodcount.total_energy += maxfood.food_grow;
    }
}

pub fn food_despawn(
    query: Query<(Entity, &FoodEnergy), Changed<FoodEnergy>>,
    mut commands: Commands,
) {
    for (entity, fe) in query.iter() {
        if fe.energy == 0 {
            commands.entity(entity).despawn();
        }
    }
}
