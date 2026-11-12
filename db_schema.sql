CREATE TABLE IF NOT EXISTS road_segments (
 id BIGINT PRIMARY KEY, road_name TEXT NOT NULL, latitude DOUBLE PRECISION NOT NULL,
 longitude DOUBLE PRECISION NOT NULL, pci DOUBLE PRECISION NOT NULL CHECK (pci BETWEEN 0 AND 100),
 rainfall_mm DOUBLE PRECISION NOT NULL DEFAULT 0, traffic_count INTEGER NOT NULL DEFAULT 0,
 historical_damage_count INTEGER NOT NULL DEFAULT 0, pothole_count INTEGER NOT NULL DEFAULT 0,
 crack_area_ratio DOUBLE PRECISION NOT NULL DEFAULT 0, damage_type TEXT NOT NULL DEFAULT 'none',
 severity TEXT NOT NULL DEFAULT 'none', observed_at TIMESTAMPTZ NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_road_segments_observed_at ON road_segments(observed_at);
CREATE TABLE IF NOT EXISTS predictions (
 id BIGSERIAL PRIMARY KEY, segment_id BIGINT NOT NULL REFERENCES road_segments(id),
 horizon_days INTEGER NOT NULL CHECK (horizon_days IN (7,15,30)), predicted_pci DOUBLE PRECISION NOT NULL,
 risk_score DOUBLE PRECISION NOT NULL, recommendation TEXT NOT NULL,
 model_version TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
