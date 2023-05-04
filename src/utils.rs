use num::pow::Pow;
use std::cmp::min;

pub struct Grid<T> {
    minx: f32,
    miny: f32,
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
            miny,
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

    pub(crate) fn within_bounds(&self, x: f32, y: f32) -> Option<(usize, usize)> {
        let ix = (x - self.minx) / self.dx;
        let iy = (y - self.miny) / self.dy;
        let outside_bounds = ix < 0. || ix >= (self.nx as f32) || iy < 0. || iy >= (self.ny as f32);
        if outside_bounds {
            None
        } else {
            Some((ix as usize, iy as usize))
        }
    }

    pub(crate) fn within_dist_of(
        &self,
        x: f32,
        y: f32,
        d: f32,
    ) -> impl Iterator<Item = &(f32, f32, T)> {
        let d2 = d.pow(2);
        let dnx = 1 + (d / self.dx).floor() as usize;
        let dny = 1 + (d / self.dy).floor() as usize;

        let (xrange, yrange) = if let Some((ix, iy)) = self.within_bounds(x, y) {
            let lowx = ix - min(ix, dnx);
            let lowy = iy - min(iy, dny);
            let highx = min(ix + dnx + 1, self.nx);
            let highy = min(iy + dny + 1, self.ny);
            ((lowx..highx), (lowy..highy))
        } else {
            let xrange = if x < self.minx {
                0..min(dnx + 1, self.nx)
            } else {
                debug_assert!(x > self.minx + (self.nx as f32) * self.dx);
                self.nx - min(dnx, self.nx)..self.nx
            };
            let yrange = if y < self.miny {
                (0..min(dny + 1, self.ny))
            } else {
                debug_assert!(x > self.miny + (self.ny as f32) * self.dy);
                (self.ny - min(dny, self.ny)..self.ny)
            };
            (xrange, yrange)
        };

        xrange
            .flat_map(move |dix| yrange.clone().map(move |diy| (dix, diy)))
            .flat_map(move |(dix, diy)| {
                debug_assert!(dix < self.nx, "dix={} >= {}", dix, self.nx);
                debug_assert!(diy < self.ny, "diy={} >= {}", diy, self.ny);

                self.dat[(diy * self.nx) + dix].iter()
            })
            .chain(self.outside.iter())
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
        debug_assert_eq!(grid.within_bounds(0.5, 0.5), Some((0, 0)));
        debug_assert_eq!(grid.within_bounds(-0.5, 0.5), None);
        debug_assert_eq!(grid.within_bounds(0.5, -0.5), None);

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
