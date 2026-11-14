"""Pydantic schemas for student feature inputs and prediction outputs."""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """Input payload for estimating student pass/fail outcome."""
    student_identifier: Optional[str] = Field(default=None, description="Pseudonymous student ID")
    school: Optional[str] = Field(default="Amity University Kolkata", description="University name (e.g. Amity University Kolkata, Adamas University, Brainware University, St. Xavier's University, Techno India University)")
    sex: Optional[Literal["F", "M"]] = Field(default="F", description="Student sex")
    age: Optional[int] = Field(default=17, ge=12, le=30, description="Student age")
    address: Optional[Literal["U", "R"]] = Field(default="U", description="Home address type (U: urban, R: rural)")
    famsize: Optional[Literal["GT3", "LE3"]] = Field(default="GT3", description="Family size (GT3: >3, LE3: <=3)")
    Pstatus: Optional[Literal["T", "A"]] = Field(default="T", description="Parent cohabitation status (T: together, A: apart)")
    Medu: Optional[int] = Field(default=2, ge=0, le=4, description="Mother's education level (0 to 4)")
    Fedu: Optional[int] = Field(default=2, ge=0, le=4, description="Father's education level (0 to 4)")
    Mjob: Optional[Literal["teacher", "health", "services", "at_home", "other"]] = Field(default="other", description="Mother's job")
    Fjob: Optional[Literal["teacher", "health", "services", "at_home", "other"]] = Field(default="other", description="Father's job")
    reason: Optional[Literal["home", "reputation", "course", "other"]] = Field(default="course", description="Reason to choose this school")
    guardian: Optional[Literal["mother", "father", "other"]] = Field(default="mother", description="Student's guardian")
    traveltime: Optional[int] = Field(default=1, ge=1, le=4, description="Home to school travel time (1: <15m, 2: 15-30m, 3: 30m-1h, 4: >1h)")
    studytime: Optional[int] = Field(default=2, ge=1, le=4, description="Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    failures: Optional[int] = Field(default=0, ge=0, le=4, description="Number of past class failures")
    schoolsup: Optional[Literal["yes", "no"]] = Field(default="no", description="Extra educational support")
    famsup: Optional[Literal["yes", "no"]] = Field(default="no", description="Family educational support")
    paid: Optional[Literal["yes", "no"]] = Field(default="no", description="Extra paid classes within the course subject")
    activities: Optional[Literal["yes", "no"]] = Field(default="no", description="Extracurricular activities")
    nursery: Optional[Literal["yes", "no"]] = Field(default="yes", description="Attended nursery school")
    higher: Optional[Literal["yes", "no"]] = Field(default="yes", description="Wants to take higher education")
    internet: Optional[Literal["yes", "no"]] = Field(default="yes", description="Internet access at home")
    romantic: Optional[Literal["yes", "no"]] = Field(default="no", description="With a romantic relationship")
    famrel: Optional[int] = Field(default=4, ge=1, le=5, description="Quality of family relationships (1 to 5)")
    freetime: Optional[int] = Field(default=3, ge=1, le=5, description="Free time after school (1 to 5)")
    goout: Optional[int] = Field(default=3, ge=1, le=5, description="Going out with friends (1 to 5)")
    Dalc: Optional[int] = Field(default=1, ge=1, le=5, description="Workday alcohol consumption (1 to 5)")
    Walc: Optional[int] = Field(default=1, ge=1, le=5, description="Weekend alcohol consumption (1 to 5)")
    health: Optional[int] = Field(default=4, ge=1, le=5, description="Current health status (1 to 5)")
    absences: Optional[int] = Field(default=4, ge=0, le=100, description="Number of school absences")
    # Timing-sensitive prior grades
    G1: Optional[float] = Field(default=None, ge=0, le=20, description="First-period grade (0-20), only eligible in late-term scenario")
    G2: Optional[float] = Field(default=None, ge=0, le=20, description="Second-period grade (0-20), only eligible in late-term scenario")


class FactorExplanation(BaseModel):
    feature: str
    value: str
    impact: str
    direction: Literal["positive", "negative"]


class PredictionResponse(BaseModel):
    student_identifier: str
    timing_scenario: str
    model_used: str
    predicted_outcome: Literal["Pass", "Fail"]
    pass_probability: float
    fail_probability: float
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"]
    recommendation: str
    top_factors: List[FactorExplanation]
    disclaimer: str


class SimulationRequest(BaseModel):
    student: StudentInput
    interventions: Dict[str, Any]
    timing_scenario: Optional[Literal["early", "late"]] = Field(default="early")
    model_name: Optional[str] = Field(default=None)


class SimulationResponse(BaseModel):
    baseline: PredictionResponse
    simulated: PredictionResponse
    probability_delta: float
    percentage_gain: float
    risk_migration: str
    improved: bool

