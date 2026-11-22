// Sokoban solver — push-optimal A* over pushes, room by room.
//
// One program answers every instance of a set in a single run. A level is a
// hallway with rooms off it (see the competition's generator); a box pushed
// out of a room's door into the hallway can never come back, so every box
// stays in its room and the level's push optimum is the sum of its rooms'
// optima. This solver therefore splits a level into its rooms and solves
// each room on its own, push-optimally, then walks the player from room to
// room and concatenates the moves.
//
// Inside a room the search is A* over pushes, not steps: a state is the
// boxes' squares plus the player's reachable region (named by its lowest
// square, since any square of the region is as good as any other for the
// next push), each push costs 1, and the estimate is the minimum-cost
// matching of boxes to goals on precomputed walking distances (admissible:
// one push moves one box one square). Dead squares (a box there can never
// reach a goal, found by pulling boxes back from the goals) and frozen
// 2x2 blocks are pruned; neither prunes a solvable state, so the first
// solved state A* pops is push-optimal for the room.
//
// Everything is bounded by work, never by the clock: no Date, no random
// numbers, no environment reads feed the search, so two runs on one set
// write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work caps: expansions, never seconds, so runs stay deterministic. */
const ROOM_EXPANSIONS = 300_000;
const WHOLE_LEVEL_EXPANSIONS = 300_000;

/** The four moves: [dx, dy, step letter, push letter]. */
const DIRS: readonly (readonly [number, number, string, string])[] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
] as const;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: # wall, space floor, . goal, $ box, * box on a goal, @ player, + player on a goal. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The level written back as XSB (unused by the solver, kept for debugging). */
function formatLevel(level: Level): string {
  const goal = new Set(level.goals);
  const box = new Set(level.boxes);
  const lines: string[] = [];
  for (let y = 0; y < level.h; y++) {
    let line = '';
    for (let x = 0; x < level.w; x++) {
      const i = y * level.w + x;
      if (level.wall[i]) line += '#';
      else if (box.has(i)) line += goal.has(i) ? '*' : '$';
      else if (i === level.player) line += goal.has(i) ? '+' : '@';
      else line += goal.has(i) ? '.' : ' ';
    }
    lines.push(line);
  }
  return lines.join('\n') + '\n';
}

/**
 * The squares the player reaches from `from` around walls and the boxes in
 * `occupied`, marking them in `seen` with `stamp`; returns the region's
 * lowest square, which names the region in a state's key.
 */
function reach(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * The squares from which a box alone can be pushed onto some goal: pull
 * boxes back from the goals over open floor. A box pushed onto any other
 * square is dead, so the search never puts one there.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Walking distances from every goal over the floor (walls only, boxes
 * ignored): dist[g][square]. One push moves one box one square, so the
 * distance from a box to its goal is a lower bound on the pushes that box
 * still needs.
 */
function goalDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  const out: Int32Array[] = [];
  for (const g of goals) {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

/**
 * The minimum-cost matching of boxes to goals on the precomputed
 * distances: the cheapest way to pair every box with a distinct goal.
 * Admissible, and far tighter than each box's nearest goal, since two boxes
 * cannot share one goal. Brute-forced over permutations (at most 4 boxes
 * a room); for larger sets it falls back to the sum of nearest goals,
 * which stays admissible.
 */
function matchingEstimate(boxes: readonly number[], goals: readonly number[], dist: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 4 || goals.length !== n) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let g = 0; g < goals.length; g++) {
        const d = dist[g]![b]!;
        if (d >= 0 && d < best) best = d;
      }
      sum += best === Infinity ? 0 : best;
    }
    return sum;
  }
  // Every permutation of the goals, cheapest first by construction of the loop order.
  const order = goals.map((_, i) => i);
  let best = Infinity;
  const permute = (k: number, acc: number): void => {
    if (acc >= best) return;
    if (k === n) {
      best = acc;
      return;
    }
    for (let i = k; i < n; i++) {
      const t = order[k]!;
      order[k] = order[i]!;
      order[i] = t;
      const d = dist[order[k]!]![boxes[k]!]!;
      if (d >= 0) permute(k + 1, acc + d);
      const t2 = order[k]!;
      order[k] = order[i]!;
      order[i] = t2;
    }
  };
  permute(0, 0);
  return best === Infinity ? 0 : best;
}

/**
 * Whether the box just pushed to `t` sits in a frozen 2x2 block: four
 * squares of walls and boxes with a box among them off its goal. Nothing
 * in the block can ever move again, so the state is dead.
 */
function frozenBlock(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
    if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if (
      (occupied[a] && !goal[a]) ||
      (occupied[b] && !goal[b]) ||
      (occupied[c] && !goal[c]) ||
      (occupied[d] && !goal[d])
    ) {
      return true;
    }
  }
  return false;
}

/** A state's key: the region's lowest square, then the boxes' squares, as digits in base `size` (exact while size ** (boxes + 1) stays under 2 ** 53: five boxes or fewer on any board this generator draws); a string past that, so a big whole-level search never collides two states. */
function keyOf(boxes: readonly number[], low: number, size: number): number | string {
  if (boxes.length > 5) return low + ':' + boxes.join(',');
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Solved {
  pushes: number;
  moves: string;
  expansions: number;
}

/**
 * The push-optimal solution of `level` (or null when the search gives up
 * inside `maxExpansions` states). A* over pushes: a state is the boxes plus
 * the player's region, each push costs 1, the estimate is the matching
 * bound above. The open list is a heap on (f, deeper first, older first),
 * so ties break the same way on every run. The first solved state popped
 * is optimal: the estimate never overstates the pushes left, and the
 * pruning never cuts a solvable state.
 */
function solvePushOptimal(level: Level, maxExpansions: number): Solved | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  const dist = goalDistances(level);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (boxes: readonly number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;

  const statesBoxes: number[][] = [level.boxes.slice()];
  const statesPlayer: number[] = [level.player];
  const statesG: number[] = [0];
  const statesF: number[] = [matchingEstimate(level.boxes, goals, dist)];
  // Each record's key: a key first reached on a worse path leaves a stale
  // record behind when a better path to it is found, and the pop step
  // skips those by comparing the record's g against the key's best g.
  const statesKey: (number | string)[] = [];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1];
  const best = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const initialKey = keyOf(level.boxes, reach(w, wall, occupied, level.player, seen, ++stamp, stack), size);
  statesKey.push(initialKey);
  best.set(initialKey, 0);
  for (const b of level.boxes) occupied[b] = 0;

  // Heap of state indices, ordered by f, then deeper g, then older index.
  const heap: number[] = [0];
  const less = (a: number, b: number): boolean =>
    statesF[a]! < statesF[b]! ||
    (statesF[a]! === statesF[b]! &&
      (statesG[a]! > statesG[b]! || (statesG[a]! === statesG[b]! && a < b)));
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  let expansions = 0;
  let found = solved(level.boxes) ? 0 : -1;
  while (heap.length > 0 && found < 0) {
    const s = heapPop();
    // A record left stale by a better path to its key: skip it. Without
    // this, a key first reached on a worse path would stand, and the
    // search could answer over the optimum.
    if (best.get(statesKey[s]!) !== statesG[s]!) continue;
    const boxes = statesBoxes[s]!;
    const g = statesG[s]!;
    // A solved state is optimal only when popped, never when first
    // generated: the heap pops in f order, and a solved state's f is its
    // push count, so the first solved state popped is the cheapest.
    if (solved(boxes)) {
      found = s;
      break;
    }
    // The region this state stands in, stamped `here`; the player can push any box it can stand behind.
    for (const b of boxes) occupied[b] = 1;
    reach(w, wall, occupied, statesPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    expansions++;
    if (expansions > maxExpansions) {
      for (const b of boxes) occupied[b] = 0;
      return null;
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(w, wall, occupied, b, mark, ++stamp, stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const ng = g + 1;
        // A key reached again on a better path reopens it: the old
        // record stays in the heap but pops as stale. Skipping the
        // re-generation instead would freeze the worse cost in.
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        const idx = statesBoxes.length;
        best.set(key, ng);
        statesBoxes.push(next);
        statesPlayer.push(b);
        statesG.push(ng);
        statesF.push(ng + matchingEstimate(next, goals, dist));
        statesKey.push(key);
        parent.push(s);
        pushFrom.push(b);
        pushDir.push(di);
        heapPush(idx);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const di = pushDir[s]!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += DIRS[di]![3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves, expansions };
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * The level's rooms: cut the dead squares of the player's row (the
 * hallway: a box there reaches no goal) out of the floor, and each part
 * left that holds a box is a room. Null when some part holds more or fewer
 * boxes than goals, when a goal falls outside every part, or when a box
 * starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in
 * turn, the other rooms' boxes standing as walls, the player walking on
 * from where the last room left it. A box never leaves its room, so the
 * rooms' optima add up to the level's. Falls back to one search over the
 * whole level when the level is no set of rooms, and to an empty answer
 * when even that gives up: an empty answer is valid, worth no points,
 * but never breaks the run.
 */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.every((b) => level.goals.includes(b))) return '';
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    const done = new Set<Room>();
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        // A solved room's boxes stand on its goals now; an unsolved room's
        // still stand where the level put them. Walling off the wrong
        // squares would seal the player in, or let a walk cross a box.
        for (const b of done.has(other) ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = solvePushOptimal(sub, ROOM_EXPANSIONS);
      if (solved === null) {
        const whole = solvePushOptimal(level, WHOLE_LEVEL_EXPANSIONS);
        return whole === null ? '' : whole.moves;
      }
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      done.add(room);
    }
    return moves;
  }
  const whole = solvePushOptimal(level, WHOLE_LEVEL_EXPANSIONS);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}
