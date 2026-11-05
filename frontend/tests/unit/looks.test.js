import { describe, expect, test } from 'bun:test';
import { LOOKS, normalizeLook } from '../../src/looks.js';
import {
  createRenderer, makeGaussianKernel, blurNormalized, dogFromBlurs, sobelWithAngle,
} from '../../src/renderer.js';
import { fillCodes, codesToText } from '../../src/ascii.js';
import { unsharpMask } from '../../src/effects.js';

const neutral = { id: 'original', autoContrast: false, midtone: 1, sharpness: 0, invert: false };

function originalStages(gray, width, height) {
  const length = gray.length;
  const unit = Float64Array.from(gray, (value) => value / 255);
  const scratch = new Float64Array(length);
  const blur1 = new Float64Array(length);
  const blur2 = new Float64Array(length);
  blurNormalized(unit, width, height, makeGaussianKernel(1), blur1, scratch);
  blurNormalized(unit, width, height, makeGaussianKernel(2), blur2, scratch);
  const dog = dogFromBlurs(blur1, blur2, new Uint8Array(length));
  const sobel = sobelWithAngle(dog, width, height, {
    gx: new Float64Array(length), gy: new Float64Array(length),
    angles: new Float64Array(length), magnitude: new Uint8Array(length),
  });
  const shadeCodes = new Uint8Array(length);
  const edgeCodes = new Uint8Array(length);
  fillCodes(gray, sobel.magnitude, sobel.angles, shadeCodes, edgeCodes, 90);
  return {
    width, height, text: codesToText(shadeCodes, edgeCodes, width, height),
    gray, blur1, blur2, dog, ...sobel, shadeCodes, edgeCodes,
  };
}

describe('strict normalized look state', () => {
  test('defaults are neutral and every recipe supplies a complete neutral state', () => {
    expect(normalizeLook()).toEqual(neutral);
    expect(LOOKS.map(({ id }) => id)).toEqual(['original', 'photo', 'print', 'outline']);
    for (const recipe of LOOKS) {
      expect(normalizeLook(recipe)).toEqual({ ...neutral, id: recipe.id });
      expect(recipe.edges).toBe(recipe.id === 'original' || recipe.id === 'outline');
    }
    const candidate = { id: 'photo', autoContrast: true, midtone: 0.5, sharpness: 1.5, invert: true };
    const normalized = normalizeLook(candidate);
    expect(normalized).toEqual(candidate);
    expect(normalized).not.toBe(candidate);
    candidate.id = 'outline';
    expect(normalized.id).toBe('photo');
  });

  test('unknown IDs, wrong types, explicit undefined and invalid bounds fail', () => {
    for (const invalid of [null, [], 'photo', 1,
      { id: 'Photo' }, { id: 'unknown' }, { id: 1 }, { id: undefined },
      { autoContrast: 1 }, { invert: 'false' }, { invert: undefined },
      { midtone: '1' }, { midtone: NaN }, { midtone: Infinity },
      { midtone: 0.49 }, { midtone: 2.01 }, { sharpness: -0.1 },
      { sharpness: 1.51 }, { sharpness: null }]) {
      expect(() => normalizeLook(invalid)).toThrow();
    }
    expect(normalizeLook({ midtone: 2, sharpness: 0 })).toEqual({ ...neutral, midtone: 2 });
    expect(() => createRenderer(2, 2, {}, { id: 'unknown' })).toThrow('Unknown look');
  });
});

describe('look renderer stages', () => {
  test('neutral Original keeps every existing snapshot plane and text exactly', () => {
    const width = 7;
    const height = 5;
    const input = Uint8Array.from({ length: width * height }, (_, i) => (i * 73 + 19) % 256);
    const expected = originalStages(input, width, height);
    for (const look of [undefined, neutral]) {
      const renderer = createRenderer(width, height, {}, look);
      expect(renderer.renderGray(input)).toBe(expected.text);
      expect(renderer.snapshot()).toEqual(expected);
      expect(renderer.renderGray(input)).toBe(expected.text);
    }
  });

  test('snapshots copy on request and cannot mutate subsequent output', () => {
    const renderer = createRenderer(2, 2);
    const input = Uint8Array.from([10, 60, 180, 255]);
    const text = renderer.renderGray(input);
    const saved = renderer.snapshot();
    saved.gray.fill(0);
    saved.dog.fill(255);
    saved.shadeCodes.fill(32);
    expect(renderer.snapshot().gray).toEqual(input);
    expect(renderer.renderGray(input)).toBe(text);
    expect(input).toEqual(Uint8Array.from([10, 60, 180, 255]));
  });

  test('Photo and Print need no edge planes and reuse clean shading workspaces', () => {
    const input = Uint8Array.from({ length: 16 }, (_, i) => (i * 47) % 256);
    for (const id of ['photo', 'print']) {
      const renderer = createRenderer(4, 4, {}, { id });
      const first = renderer.renderGray(input);
      const snapshot = renderer.snapshot();
      for (const key of ['blur1', 'blur2', 'dog', 'gx', 'gy', 'magnitude', 'angles']) {
        expect(snapshot[key]).toBeNull();
      }
      expect(snapshot.edgeCodes).toEqual(new Uint8Array(16).fill(32));
      expect(renderer.renderGray(new Uint8Array(16).fill(255))).toBe('@@@@\n@@@@\n@@@@\n@@@@');
      expect(renderer.renderGray(input)).toBe(first);
      expect(renderer.snapshot().shadeCodes).toEqual(snapshot.shadeCodes);
    }
  });

  test('Outline removes interior shades and applies sharpness before contours', () => {
    const flat = new Uint8Array(9).fill(255);
    expect(createRenderer(3, 3, {}, { id: 'outline' }).renderGray(flat)).toBe('   \n   \n   ');
    expect(createRenderer(3, 3).renderGray(flat)).toBe('@@@\n@@@\n@@@');
    const width = 9;
    const height = 9;
    const input = new Uint8Array(width * height).fill(100);
    input[40] = 200;
    const enhanced = input.slice();
    unsharpMask(enhanced, width, height, 1, new Float64Array(input.length));
    const expected = originalStages(enhanced, width, height);
    const renderer = createRenderer(width, height, {}, { id: 'outline', sharpness: 1 });
    renderer.renderGray(input);
    const result = renderer.snapshot();
    expect(result.gray).toEqual(enhanced);
    expect(result.gray).not.toEqual(input);
    expect(result.dog).toEqual(expected.dog);
    expect(result.dog).not.toEqual(originalStages(input, width, height).dog);
    expect(result.angles).toEqual(expected.angles);
    expect(result.edgeCodes).toEqual(expected.edgeCodes);
    expect(result.shadeCodes).toEqual(new Uint8Array(input.length).fill(32));
    expect(result.text).toBe(codesToText(result.shadeCodes, expected.edgeCodes, width, height));
  });

  test('tone precedes shading and edges, and reselecting neutral Original restores exact text', () => {
    const input = Uint8Array.from([0, 64, 128, 255]);
    const original = createRenderer(2, 2).renderGray(input);
    for (const id of ['original', 'photo', 'print', 'outline']) {
      const adjusted = createRenderer(2, 2, {}, { id, midtone: 2, invert: true });
      adjusted.renderGray(input);
      expect(adjusted.snapshot().gray).toEqual(Uint8Array.from([255, 239, 191, 0]));
      expect(input).toEqual(Uint8Array.from([0, 64, 128, 255]));
    }
    expect(createRenderer(2, 2, {}, LOOKS[0]).renderGray(input)).toBe(original);
  });
});
