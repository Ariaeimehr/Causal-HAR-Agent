export interface SensorNode {
  id: string;
  anatomical_location: string;
  sensor_type: string;
  axis: string;
  description: string;
}

export interface SensorEdge {
  source: string;
  target: string;
  confidence: number;
  mechanism: string;
  time_delay_ms?: number;
}

export interface CausalGraphData {
  activity: string;
  reasoning_summary: string;
  nodes: SensorNode[];
  edges: SensorEdge[];
  is_dag?: boolean;
}

export interface ChannelStats {
  mean: number;
  std: number;
  variance: number;
  min: number;
  max: number;
  peak_to_peak: number;
  rms: number;
  dominant_frequency_hz: number;
  spectral_energy: number;
  spectral_entropy: number;
}

export interface FeaturePayload {
  sampling_rate_hz: number;
  channel_features: Record<string, ChannelStats>;
  cross_correlation_matrix: Record<string, Record<string, number>>;
}

export interface RawSensorStream {
  time: number[];
  channels: string[];
  data: Record<string, number[]>;
  samplingRate: number;
}

export interface RobustnessBenchmarkResult {
  clean: {
    baselineAcc: number;
    causalAcc: number;
  };
  currentPerturbation: {
    noiseSigma: number;
    dropoutRate: number;
    misalignmentDeg: number;
    baselineAcc: number;
    causalAcc: number;
    deltaAdvantage: number;
    baselineF1: number;
    causalF1: number;
    relativeDropBaseline: number;
    relativeDropCausal: number;
  };
  sweeps: {
    noise: Array<{ sigma: number; baselineAcc: number; causalAcc: number }>;
    dropout: Array<{ dropoutRate: number; baselineAcc: number; causalAcc: number }>;
  };
}
