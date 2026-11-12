from datetime import datetime
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base
class RoadSegment(Base):
    __tablename__='road_segments'
    id:Mapped[int]=mapped_column(BigInteger,primary_key=True)
    road_name:Mapped[str]=mapped_column(String,nullable=False)
    latitude:Mapped[float]=mapped_column(Float,nullable=False)
    longitude:Mapped[float]=mapped_column(Float,nullable=False)
    pci:Mapped[float]=mapped_column(Float,nullable=False)
    rainfall_mm:Mapped[float]=mapped_column(Float,default=0)
    traffic_count:Mapped[int]=mapped_column(Integer,default=0)
    historical_damage_count:Mapped[int]=mapped_column(Integer,default=0)
    pothole_count:Mapped[int]=mapped_column(Integer,default=0)
    crack_area_ratio:Mapped[float]=mapped_column(Float,default=0)
    damage_type:Mapped[str]=mapped_column(String,default='none')
    severity:Mapped[str]=mapped_column(String,default='none')
    observed_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
class Prediction(Base):
    __tablename__='predictions'
    id:Mapped[int]=mapped_column(BigInteger,primary_key=True,autoincrement=True)
    segment_id:Mapped[int]=mapped_column(ForeignKey('road_segments.id'),nullable=False)
    horizon_days:Mapped[int]=mapped_column(Integer,nullable=False)
    predicted_pci:Mapped[float]=mapped_column(Float,nullable=False)
    risk_score:Mapped[float]=mapped_column(Float,nullable=False)
    recommendation:Mapped[str]=mapped_column(String,nullable=False)
    model_version:Mapped[str]=mapped_column(String,nullable=False)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
