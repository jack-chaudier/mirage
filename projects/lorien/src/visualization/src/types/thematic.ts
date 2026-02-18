export const THEMATIC_AXES = [
  'order_chaos',
  'truth_deception',
  'loyalty_betrayal',
  'innocence_corruption',
  'freedom_control'
] as const;

export type ThematicAxis = (typeof THEMATIC_AXES)[number];
