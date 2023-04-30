use num::pow::Pow;
use std::cmp::{min};

pub enum IterEither<TA, TB> {
    A(TA),
    B(TB),
}

impl<T, TA, TB> Iterator for IterEither<TA, TB>
where
    TA: Iterator<Item = T>,
    TB: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IterEither::A(a) => a.next(),
            IterEither::B(b) => b.next(),
        }
    }
}

pub struct Grid<T> {
    minx: f32,
    maxx: f32,
    miny: f32,
    maxy: f32,
    dx: f32,
    dy: f32,
    nx: usize,
    ny: usize,
    dat: Vec<Vec<(f32, f32, T)>>,
    outside: Vec<(f32, f32, T)>,
}

impl<T> Grid<T> {
    pub(crate) fn new(minx: f32, maxx: f32, miny: f32, maxy: f32, nx: usize, ny: usize) -> Self {
        let dx = (maxx - minx) / (nx as f32);
        let dy = (maxy - miny) / (ny as f32);
        Self {
            minx,
            maxx,
            miny,
            maxy,
            dx,
            dy,
            nx,
            ny,
            dat: (0..nx * ny).map(|_| vec![]).collect(),
            outside: vec![],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.outside.clear();
        self.dat.iter_mut().for_each(|v| v.clear());
    }

    pub(crate) fn insert(&mut self, x: f32, y: f32, t: T) {
        let ix = (x - self.minx) / self.dx;
        let iy = (y - self.miny) / self.dy;
        if ix < 0. || ix >= (self.nx as f32) || iy < 0. || iy >= (self.ny as f32) {
            self.outside.push((x, y, t))
        } else {
            let ix = ix.floor() as usize;
            let iy = iy.floor() as usize;
            let index = iy * self.nx + ix;
            self.dat[index].push((x, y, t))
        }
    }

    pub(crate) fn within_dist_of(
        &self,
        x: f32,
        y: f32,
        d: f32,
    ) -> impl Iterator<Item = &(f32, f32, T)> {
        let d2 = d.pow(2);
        let ix = (x - self.minx) / self.dx;
        let iy = (y - self.miny) / self.dy;
        let dnx = (d / self.dx).floor() as usize;
        let dny = (d / self.dy).floor() as usize;
        let outside_bounds = ix < 0. || ix >= (self.nx as f32) || iy < 0. || iy >= (self.ny as f32);

        let within_bounds = if !outside_bounds {
            let ix = ix as usize;
            let iy = iy as usize;
            debug_assert!(ix < self.nx);
            debug_assert!(iy < self.ny);
            let lowx = ix - min(ix, dnx + 1);
            let lowy = iy - min(iy, dny + 1);
            let highx = min(ix + dnx + 1, self.nx);
            let highy = min(iy + dny + 1, self.ny);
            let within_bounds = (lowx..highx)
                .flat_map(move |dix| (lowy..highy).map(move |diy| (dix, diy)))
                .flat_map(move |(dix, diy)| {
                    debug_assert!(dix < self.nx, "dix={} >= {}", dix, self.nx);
                    debug_assert!(diy < self.ny, "diy={} >= {}", diy, self.ny);

                    self.dat[(diy * self.nx) + dix].iter()
                });
            IterEither::A(within_bounds)
        } else {
            let all_possible = self.dat.iter().flat_map(|v| v.iter());
            IterEither::B(all_possible)
        };
        let outside_bounds = self.outside.iter();

        within_bounds
            .chain(outside_bounds)
            .filter(move |(tx, ty, _)| (tx - x).pow(2) + (ty - y).pow(2) <= d2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_simple() {
        let mut grid = Grid::new(0.0, 1.0, 0.0, 1.0, 1, 1);
        grid.insert(0.5, 0.1, ());
        grid.insert(0.5, 0.99, ());

        let a = grid.within_dist_of(0.5, 0.5, 0.55).collect::<Vec<_>>();
        assert_eq!(a.len(), 2, "a={:?}", a);
        let a = grid.within_dist_of(0.5, 0.5, 0.45).collect::<Vec<_>>();
        assert_eq!(a.len(), 1, "a={:?}", a);
        let a = grid.within_dist_of(0.5, 0.5, 0.3).collect::<Vec<_>>();
        assert!(a.is_empty(), "a={:?}", a);
    }
    #[test]
    fn test_outside() {
        let mut grid = Grid::new(0.0, 1.0, 0.0, 1.0, 1, 1);
        grid.insert(0.5, 0.1, ());
        grid.insert(0.5, 0.99, ());
        grid.insert(0.5, 1.01, ());

        let a = grid.within_dist_of(0.5, 0.5, 0.55).collect::<Vec<_>>();
        assert_eq!(a.len(), 3, "a={:?}", a);
        let a = grid.within_dist_of(0.5, 0.5, 0.5).collect::<Vec<_>>();
        assert_eq!(a.len(), 2, "a={:?}", a);
        let a = grid.within_dist_of(0.5, 0.5, 0.45).collect::<Vec<_>>();
        assert_eq!(a.len(), 1, "a={:?}", a);
        let a = grid.within_dist_of(0.5, 0.5, 0.3).collect::<Vec<_>>();
        assert!(a.is_empty(), "a={:?}", a);
    }

    #[test]
    fn test_border() {
        let mut grid = Grid::new(0.0, 1.0, 0.0, 1.0, 2, 2);
        grid.insert(0.49, 0.49, ());
        grid.insert(0.51, 0.51, ());

        let a = grid.within_dist_of(0.49, 0.51, 0.1).collect::<Vec<_>>();
        assert_eq!(a.len(), 2, "a={:?}", a);
    }
}
