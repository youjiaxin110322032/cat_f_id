from sqlalchemy import Column, Integer, String, ForeignKey, Text, Table
from sqlalchemy.orm import relationship
from database import Base

cat_caretaker = Table(
    "cat_caretaker", Base.metadata,
    Column("cat_id", Integer, ForeignKey("cats.id"), primary_key=True),
    Column("caretaker_id", Integer, ForeignKey("caretakers.id"), primary_key=True),
)

class Household(Base):
    __tablename__ = "households"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    address = Column(String)
    cats = relationship("Cat", back_populates="household")

class Caretaker(Base):
    __tablename__ = "caretakers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    phone = Column(String)
    cats = relationship("Cat", secondary=cat_caretaker, back_populates="caretakers")

class Cat(Base):
    __tablename__ = "cats"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    sex = Column(String, default="U")
    birth_year = Column(Integer)
    coat = Column(String)
    ear_tip = Column(Integer, default=0)
    household_id = Column(Integer, ForeignKey("households.id"))
    household = relationship("Household", back_populates="cats")
    caretakers = relationship("Caretaker", secondary=cat_caretaker, back_populates="cats")
