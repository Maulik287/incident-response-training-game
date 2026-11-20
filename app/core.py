"""Incident Response Training Game - Core Engine and Milestones.

Public service trainees play a simulation game where they predict incident
sequences (e.g., traffic_collision -> road_blockage -> emergency_medical)
and allocate emergency resources to learn optimal response strategies.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Try importing thilimarketbasket
try:
    from thilimarketbasket.mine import frequent_itemsets, rules
except ImportError:
    frequent_itemsets = None  # type: ignore
    rules = None  # type: ignore

# ============================================================================
# Domain Constants & Taxonomy
# ============================================================================

INCIDENT_TYPES = [
    "traffic_collision",
    "road_blockage",
    "fuel_leak",
    "vehicle_fire",
    "emergency_medical",
    "structural_damage",
    "crowd_surge",
    "power_outage",
    "weather_hazard",
    "hazmat_spill",
    "secondary_collision",
    "police_perimeter",
]

RESOURCE_TYPES = ["police", "fire", "ambulance", "traffic"]

# Standardized resource demand for full mitigation per incident type
INCIDENT_RESOURCE_REQUIREMENTS: Dict[str, Dict[str, int]] = {
    "traffic_collision": {"police": 2, "traffic": 1},
    "road_blockage": {"traffic": 2, "police": 1},
    "fuel_leak": {"fire": 2, "traffic": 1},
    "vehicle_fire": {"fire": 3, "police": 1, "ambulance": 1},
    "emergency_medical": {"ambulance": 2, "police": 1},
    "structural_damage": {"traffic": 2, "police": 1},
    "crowd_surge": {"police": 3},
    "power_outage": {"traffic": 2, "police": 1},
    "weather_hazard": {"traffic": 2},
    "hazmat_spill": {"fire": 3, "police": 2, "ambulance": 1},
    "secondary_collision": {"ambulance": 2, "police": 2, "traffic": 1},
    "police_perimeter": {"police": 2},
}

# ============================================================================
# Milestone 1: Model public incident reports as ordered sequences of events
# ============================================================================

@dataclass
class IncidentEvent:
    """A discrete incident event occurring at a specific point in time."""
    event_type: str
    minute_offset: int
    description: str
    severity: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentReport:
    """Public incident report representing a real-world multi-stage emergency."""
    report_id: str
    title: str
    location: str
    timestamp: str
    events: List[IncidentEvent]

    @property
    def ordered_sequence(self) -> List[str]:
        """Chronologically ordered sequence of incident types."""
        sorted_events = sorted(self.events, key=lambda e: e.minute_offset)
        return [e.event_type for e in sorted_events]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "location": self.location,
            "timestamp": self.timestamp,
            "events": [e.to_dict() for e in self.events],
            "ordered_sequence": self.ordered_sequence,
        }


def get_default_incident_reports() -> List[IncidentReport]:
    """Curated public incident reports reflecting diverse real-world emergency cascades."""
    raw_scenarios = [
        # 1. Highway Multi-Vehicle Collision with Fire Cascade
        ("INC-2026-001", "Interstate 95 Multi-Car Crash", "I-95 Mile 42 Northbound", "2026-03-01T08:15:00Z", [
            ("traffic_collision", 0, "Two sedans collided in fast lane during rush hour", "high"),
            ("road_blockage", 5, "Debris and disabled cars obstruct two lanes", "medium"),
            ("fuel_leak", 12, "Ruptured gas tank spilling gasoline onto highway asphalt", "high"),
            ("vehicle_fire", 18, "Fuel ignited by exhaust spark, engulfing one vehicle", "critical"),
            ("emergency_medical", 22, "Severe burns and smoke inhalation trauma dispatch", "critical"),
            ("police_perimeter", 28, "Highway closed at exit 41, detours initiated", "high"),
        ]),
        # 2. Downtown Urban Intersection Pileup
        ("INC-2026-002", "Central Metro Bus vs Delivery Van", "Market St & 4th Ave", "2026-03-02T14:30:00Z", [
            ("traffic_collision", 0, "Delivery van t-boned municipal transit bus", "high"),
            ("emergency_medical", 6, "Multiple passengers with whiplash and contusions", "high"),
            ("road_blockage", 10, "Bus stranded across double tram tracks and intersection", "medium"),
            ("crowd_surge", 15, "Over 150 pedestrians gathering at corners filming", "medium"),
            ("police_perimeter", 20, "Perimeter taped off for passenger triage zone", "medium"),
        ]),
        # 3. Industrial Chemical Corridor Spill
        ("INC-2026-003", "Tanker Truck Rollover", "Port Industrial Bypass Route 7", "2026-03-03T11:00:00Z", [
            ("traffic_collision", 0, "Chemical tanker jackknifed on wet ramp curve", "high"),
            ("hazmat_spill", 8, "Hydrochloric acid leaking from cracked tank valve", "critical"),
            ("police_perimeter", 14, "Half-mile evacuation zone and cordon established", "critical"),
            ("road_blockage", 18, "Industrial bypass completely closed both ways", "high"),
            ("emergency_medical", 25, "Driver treated for respiratory chemical exposure", "medium"),
        ]),
        # 4. Severe Weather Winter Pileup
        ("INC-2026-004", "Black Ice Chain Collision", "State Route 18 Mountain Pass", "2026-03-04T06:45:00Z", [
            ("weather_hazard", 0, "Freezing rain creates severe black ice across bridge deck", "high"),
            ("traffic_collision", 10, "SUV lost traction, struck concrete median divider", "medium"),
            ("secondary_collision", 16, "Two following vehicles rear-end initial crash in fog", "high"),
            ("road_blockage", 22, "Both westbound lanes completely impassable", "high"),
            ("emergency_medical", 27, "Ambulances dispatched for fractured limb and shock", "high"),
            ("structural_damage", 35, "Bridge barrier fractured, guardrail hanging over ravine", "high"),
        ]),
        # 5. Grid Failure & Major Signal Outage
        ("INC-2026-005", "Substation Transformer Fire and Signal Blackout", "Commerce Way & Tech Blvd", "2026-03-05T17:10:00Z", [
            ("power_outage", 0, "Grid transformer blown, 12 major traffic signals dark", "high"),
            ("traffic_collision", 12, "Uncontrolled intersection blind t-bone crash", "high"),
            ("road_blockage", 18, "Gridlock paralyzing five adjacent downtown blocks", "high"),
            ("police_perimeter", 25, "Officers deployed for manual traffic direction", "medium"),
            ("emergency_medical", 30, "Driver transport for neck and spine evaluation", "medium"),
        ]),
        # 6. Commercial Truck Fuel Fire on Arterial
        ("INC-2026-006", "Diesel Delivery Truck Crash", "Oakridge Expressway", "2026-03-06T09:20:00Z", [
            ("traffic_collision", 0, "Commercial flatbed spun into highway pillar", "high"),
            ("fuel_leak", 7, "Saddle tank rupture spilling 75 gallons of diesel", "high"),
            ("vehicle_fire", 15, "Diesel ignited along ditch embankment", "critical"),
            ("road_blockage", 20, "Dense black smoke forcing closure of all northbound lanes", "high"),
            ("police_perimeter", 24, "State patrol diverting traffic onto frontage road", "medium"),
        ]),
        # 7. Storm Flash Flood & Structural Collapse
        ("INC-2026-007", "Creek Flash Flooding and Bridge Scour", "Valley Parkway Culvert", "2026-03-07T16:00:00Z", [
            ("weather_hazard", 0, "Sudden 3-inch deluge causing rapid culvert overflow", "high"),
            ("structural_damage", 15, "Culvert retaining wall undermined and sagging", "critical"),
            ("road_blockage", 20, "Standing water 2 feet deep over asphalt", "high"),
            ("traffic_collision", 28, "Car hydroplaned into flooded drainage ditch", "medium"),
            ("emergency_medical", 35, "Rescue crew pulled driver from submerged vehicle", "high"),
        ]),
        # 8. Multi-Vehicle Highway Chain Reaction
        ("INC-2026-008", "Foggy Morning Chain Crash", "Turnpike Mile 110", "2026-03-08T07:10:00Z", [
            ("weather_hazard", 0, "Dense radiation fog reducing visibility under 50 feet", "medium"),
            ("traffic_collision", 8, "Box truck hit rear of flatbed semi-trailer", "high"),
            ("secondary_collision", 14, "Three passenger cars piled into rear of box truck", "critical"),
            ("road_blockage", 20, "Total blockage of all four lanes", "critical"),
            ("emergency_medical", 26, "Mass casualty triage trailer summoned", "critical"),
            ("police_perimeter", 34, "Full turnpike ramp closure and emergency detour", "high"),
        ]),
        # 9. Tanker Spill & Controlled Burn
        ("INC-2026-009", "Solvent Leak at Highway Interchange", "Interchange I-40 & Loop 202", "2026-03-09T13:40:00Z", [
            ("traffic_collision", 0, "Chemical transport clipped barrier during lane merge", "medium"),
            ("hazmat_spill", 10, "Flammable solvent weeping from belly piping", "critical"),
            ("fuel_leak", 18, "Solvent spreading into highway rainwater catch basin", "high"),
            ("police_perimeter", 25, "One-mile safety cordon established by highway patrol", "critical"),
            ("road_blockage", 30, "Entire interchange interchange loops shuttered", "high"),
        ]),
        # 10. City Center Parade Congestion & Stampede
        ("INC-2026-010", "Festival Overcrowding & Incident", "Grand Boulevard Civic Center", "2026-03-10T19:00:00Z", [
            ("crowd_surge", 0, "Crowd overflow from festival venue surging into street", "high"),
            ("road_blockage", 12, "Pedestrians blocking emergency ambulance corridor", "high"),
            ("police_perimeter", 18, "Riot gates and crowd control barriers placed", "high"),
            ("emergency_medical", 24, "Heat exhaustion and crush injury patients treated", "high"),
        ]),
        # 11. Heavy Wind Power Pole Down on Vehicle
        ("INC-2026-011", "Gale-Force Winds Drop Utility Pole", "Lakeshore Drive", "2026-03-11T15:20:00Z", [
            ("weather_hazard", 0, "60mph wind gusts recorded along shoreline", "medium"),
            ("power_outage", 8, "Primary 13kV utility feeder snapped and down", "high"),
            ("structural_damage", 14, "Power pole collapsed across two parked cars and curb", "high"),
            ("traffic_collision", 20, "Moving car collided with downed live line", "critical"),
            ("road_blockage", 25, "Lakeshore Drive blocked in both directions", "high"),
            ("police_perimeter", 30, "Electrical safety perimeter held until utility arrival", "high"),
        ]),
        # 12. Expressway Rear-End to Medical Dispatch
        ("INC-2026-012", "Commuter Traffic Rear-End Impact", "Sunrise Expressway Exit 8", "2026-03-12T08:30:00Z", [
            ("traffic_collision", 0, "Sudden slowdown caused 3-car accordion collision", "medium"),
            ("road_blockage", 5, "Left shoulder and median lane blocked", "medium"),
            ("emergency_medical", 12, "Driver in middle vehicle reporting severe chest pain", "high"),
            ("police_perimeter", 18, "Police escorting ambulances through shoulder lane", "medium"),
        ]),
        # 13. Overpass Impact & Structural Guardrail Failure
        ("INC-2026-013", "Oversized Load Hits Overpass Beam", "Airport Parkway", "2026-03-13T10:15:00Z", [
            ("traffic_collision", 0, "Oversized crane struck bottom flange of railway overpass", "high"),
            ("structural_damage", 8, "Concrete spalling and damaged steel girders", "critical"),
            ("road_blockage", 14, "Parkway closed beneath bridge for structural survey", "high"),
            ("police_perimeter", 20, "Detour posted around railway overpass", "medium"),
        ]),
        # 14. Gas Station Fire & Highway Smoke Hazard
        ("INC-2026-014", "Service Station Pump Fire", "State Route 4 Service Plaza", "2026-03-14T12:00:00Z", [
            ("fuel_leak", 0, "Dispenser nozzle ripped from island by departing SUV", "high"),
            ("vehicle_fire", 6, "Fuel flash fire consuming vehicle and fuel canopy", "critical"),
            ("road_blockage", 15, "Thick smoke blanket across neighboring freeway lanes", "high"),
            ("police_perimeter", 20, "Plaza evacuated and freeway off-ramps barricaded", "critical"),
            ("emergency_medical", 28, "Station attendant treated for first-degree burns", "medium"),
        ]),
        # 15. Winter Storm Multi-Car Pileup
        ("INC-2026-015", "Blizzard Ice Storm Highway Stack", "Interstate 80 Mile 200", "2026-03-15T18:00:00Z", [
            ("weather_hazard", 0, "Freezing rain transition to zero-visibility snow", "high"),
            ("traffic_collision", 12, "Tractor trailer jackknifed across icy median", "high"),
            ("secondary_collision", 18, "Five passenger vehicles slid into trailer side", "critical"),
            ("fuel_leak", 24, "Torn saddle tank leaking fuel over frozen roadway", "high"),
            ("road_blockage", 30, "Interstate 80 completely closed eastbound", "critical"),
            ("emergency_medical", 36, "Four victims extracted with hydraulic spreaders", "critical"),
            ("police_perimeter", 42, "Traffic rerouted at previous highway junction", "high"),
        ]),
        # 16. Downtown Water Main Break & Sinkhole
        ("INC-2026-016", "36-inch Water Main Rupture", "Pine St & 8th Avenue", "2026-03-16T04:30:00Z", [
            ("structural_damage", 0, "Water main collapsed roadbed creating 15ft sinkhole", "critical"),
            ("road_blockage", 10, "Intersection flooded and street collapsed", "critical"),
            ("power_outage", 18, "Flooding in underground electrical vaults shuts streetlights", "high"),
            ("police_perimeter", 25, "Barricades established 2 blocks in each direction", "high"),
            ("traffic_collision", 35, "Vehicle plunged front tires into edge of sinkhole", "medium"),
        ]),
        # 17. High-Speed Police Chase Crash
        ("INC-2026-017", "Fleeing Suspect Crash at Boulevard", "Broad St & Elm Ave", "2026-03-17T22:45:00Z", [
            ("traffic_collision", 0, "Stolen sedan lost control at 80mph, hit two parked cars", "critical"),
            ("structural_damage", 6, "Sheared fire hydrant and knocked down streetlight", "high"),
            ("emergency_medical", 11, "Two injured occupants trapped in wreckage", "critical"),
            ("crowd_surge", 18, "Bar patrons and onlookers crowd scene", "medium"),
            ("police_perimeter", 23, "Crime scene tape and tactical crowd line formed", "high"),
            ("road_blockage", 28, "Broad Street cordoned for major collision investigation", "high"),
        ]),
        # 18. Highway Hazardous Waste Barrel Spill
        ("INC-2026-018", "Flatbed Drops Chemical Drums", "Outer Beltway Mile 15", "2026-03-18T11:25:00Z", [
            ("traffic_collision", 0, "Truck sudden brake caused cargo straps to snap", "medium"),
            ("hazmat_spill", 7, "Six 55-gallon drums of industrial oxidizer burst", "critical"),
            ("road_blockage", 14, "Beltway right two lanes contaminated", "high"),
            ("police_perimeter", 20, "Hazard perimeter established upwind", "high"),
            ("emergency_medical", 30, "Hazmat decontamination team checking first responders", "medium"),
        ]),
        # 19. Electric Vehicle Battery Thermal Runaway
        ("INC-2026-019", "EV High-Speed Guardrail Crash", "State Highway 52", "2026-03-19T13:10:00Z", [
            ("traffic_collision", 0, "Electric sedan struck steel guardrail at highway speed", "high"),
            ("structural_damage", 5, "Guardrail deformed 30 feet into ditch", "medium"),
            ("vehicle_fire", 14, "Lithium-ion battery pack entered thermal runaway", "critical"),
            ("emergency_medical", 20, "Driver pulled out with spinal injury precautions", "high"),
            ("road_blockage", 28, "Copious water tanker shuttle shutting both lanes", "high"),
            ("police_perimeter", 35, "Safety cordon for toxic battery fumes", "high"),
        ]),
        # 20. Tunnel Traffic Stoppage & Fire Alert
        ("INC-2026-020", "River Tunnel Disabled Vehicle & Fire", "Central Harbor Tunnel Tube 2", "2026-03-20T17:50:00Z", [
            ("traffic_collision", 0, "Van stalled and clipped inside tunnel portal", "medium"),
            ("road_blockage", 5, "Single bore tube blocked entirely", "high"),
            ("fuel_leak", 12, "Oil and gasoline puddle spreading across tunnel deck", "high"),
            ("vehicle_fire", 18, "Small engine fire started under van hood", "critical"),
            ("emergency_medical", 24, "Tunnel evacuation with smoke inhalation cases", "critical"),
            ("police_perimeter", 30, "Tunnel portals closed, ventilation fans at maximum", "critical"),
        ]),
    ]

    reports = []
    for r_id, title, loc, ts, ev_list in raw_scenarios:
        events = [
            IncidentEvent(
                event_type=etype,
                minute_offset=moff,
                description=desc,
                severity=sev
            )
            for etype, moff, desc, sev in ev_list
        ]
        reports.append(IncidentReport(
            report_id=r_id,
            title=title,
            location=loc,
            timestamp=ts,
            events=events
        ))
    return reports


def milestone_1() -> Dict[str, Any]:
    """Milestone 1: Model public incident reports as ordered sequences of events.

    Prepares incident report data into ordered sequences of events with rich metadata,
    vocabulary mapping, and event progression validation.
    """
    reports = get_default_incident_reports()
    sequences = [r.ordered_sequence for r in reports]
    
    unique_events = sorted({ev for seq in sequences for ev in seq})
    seq_lengths = [len(seq) for seq in sequences]
    avg_length = float(np.mean(seq_lengths)) if seq_lengths else 0.0

    result = {
        "milestone": 1,
        "status": "completed",
        "description": "Public incident reports successfully modeled as ordered event sequences",
        "reports_count": len(reports),
        "unique_incident_types": unique_events,
        "vocabulary_size": len(unique_events),
        "total_event_occurrences": sum(seq_lengths),
        "average_sequence_length": round(avg_length, 2),
        "min_sequence_length": min(seq_lengths) if seq_lengths else 0,
        "max_sequence_length": max(seq_lengths) if seq_lengths else 0,
        "sample_report": reports[0].to_dict(),
        "sample_sequence": sequences[0],
    }
    return result


# ============================================================================
# Milestone 2: Discover frequent incident sequences using `thilimarketbasket`
# ============================================================================

@dataclass
class AssociationRuleResult:
    """An association rule discovered from incident sequence baskets."""
    antecedent: List[str]
    consequent: List[str]
    support: float
    confidence: float
    lift: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
        }


@dataclass
class SequentialRuleResult:
    """An ordered sequential transition rule: antecedent precedes consequent."""
    antecedent: str
    consequent: str
    support: float
    confidence: float
    lift: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
        }


def mine_incident_patterns(
    reports: Optional[List[IncidentReport]] = None,
    min_support: float = 0.15,
    min_confidence: float = 0.35,
) -> Dict[str, Any]:
    """Mine frequent incident itemsets and ordered sequence rules using thilimarketbasket."""
    if reports is None:
        reports = get_default_incident_reports()

    baskets = [r.ordered_sequence for r in reports]
    n_reports = len(baskets)

    # 1. Use thilimarketbasket for itemset mining
    if frequent_itemsets is not None:
        raw_itemsets = frequent_itemsets(baskets, min_support=min_support, max_size=4)
        raw_rules = rules(raw_itemsets, min_confidence=min_confidence) if rules is not None else []
    else:
        # Fallback pure-python itemset mining if thilimarketbasket not available
        raw_itemsets = {}
        single_counts: Dict[str, int] = {}
        for b in baskets:
            for item in set(b):
                single_counts[item] = single_counts.get(item, 0) + 1
        for item, cnt in single_counts.items():
            supp = cnt / n_reports
            if supp >= min_support:
                raw_itemsets[frozenset([item])] = supp
        raw_rules = []

    # Format frequent itemsets
    formatted_itemsets = []
    for itemset, supp in raw_itemsets.items():
        formatted_itemsets.append({
            "items": sorted(list(itemset)),
            "size": len(itemset),
            "support": round(float(supp), 4),
            "count": int(round(float(supp) * n_reports)),
        })
    formatted_itemsets.sort(key=lambda x: (-x["size"], -x["support"]))

    # Format association rules
    formatted_rules = []
    for ante, cons, supp, conf, lift in raw_rules:
        formatted_rules.append(
            AssociationRuleResult(
                antecedent=sorted(list(ante)),
                consequent=sorted(list(cons)),
                support=float(supp),
                confidence=float(conf),
                lift=float(lift),
            ).to_dict()
        )
    formatted_rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))

    # 2. Mine directed sequential transitions (temporal ordering: A precedes B)
    seq_counts: Dict[Tuple[str, str], int] = {}
    item_occ: Dict[str, int] = {}

    for b in baskets:
        seen_items = set()
        for i, first in enumerate(b):
            if first not in seen_items:
                item_occ[first] = item_occ.get(first, 0) + 1
                seen_items.add(first)
            for second in b[i+1:]:
                pair = (first, second)
                seq_counts[pair] = seq_counts.get(pair, 0) + 1

    sequential_rules = []
    for (first, second), cnt in seq_counts.items():
        supp = cnt / n_reports
        if supp >= min_support:
            conf = cnt / max(1, item_occ.get(first, 1))
            cons_prob = item_occ.get(second, 1) / n_reports
            lift = conf / max(1e-5, cons_prob)
            if conf >= min_confidence:
                sequential_rules.append(
                    SequentialRuleResult(
                        antecedent=first,
                        consequent=second,
                        support=supp,
                        confidence=conf,
                        lift=lift,
                    ).to_dict()
                )
    sequential_rules.sort(key=lambda r: (-r["confidence"], -r["lift"]))

    return {
        "n_reports": n_reports,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "total_frequent_itemsets": len(formatted_itemsets),
        "frequent_itemsets": formatted_itemsets,
        "total_association_rules": len(formatted_rules),
        "association_rules": formatted_rules,
        "total_sequential_rules": len(sequential_rules),
        "sequential_rules": sequential_rules,
    }


def milestone_2() -> Dict[str, Any]:
    """Milestone 2: Discover frequent incident sequences using `thilimarketbasket`.

    Applies frequent itemset mining and sequential pattern association to discover
    critical cascading failure chains (e.g., traffic_collision -> fuel_leak -> vehicle_fire).
    """
    patterns = mine_incident_patterns(min_support=0.15, min_confidence=0.35)
    return {
        "milestone": 2,
        "status": "completed",
        "description": "Frequent incident sequences and association rules mined with thilimarketbasket",
        "patterns_summary": patterns,
        "top_frequent_pairs": [it for it in patterns["frequent_itemsets"] if it["size"] >= 2][:5],
        "top_sequential_cascades": patterns["sequential_rules"][:5],
    }


# ============================================================================
# Milestone 3: Train an LM to predict the next likely incident given a situation
# ============================================================================

class IncidentSequenceLM:
    """Causal Language Model for Incident Sequences with Continued Pretraining.

    Models next-incident probability distributions P(incident_{t+1} | incident_1..t)
    via an autoregressive transition architecture with cross-entropy loss, perplexity
    tracking, and continued pretraining weight updates.
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        context_window: int = 4,
        smoothing: float = 0.05,
        seed: int = 42,
    ):
        self.vocab = sorted(list(set(vocab or INCIDENT_TYPES)))
        self.special_tokens = ["<BOS>", "<EOS>", "<UNK>"]
        self.all_tokens = self.special_tokens + self.vocab
        self.token2id = {tok: idx for idx, tok in enumerate(self.all_tokens)}
        self.id2token = {idx: tok for idx, tok in enumerate(self.all_tokens)}
        self.vocab_size = len(self.all_tokens)
        self.context_window = context_window
        self.smoothing = smoothing
        self.rng = np.random.default_rng(seed)

        # Logit transition parameters: [context_depth, vocab_size, vocab_size]
        # represents direct bigram, trigram, and sequential attention weights
        self.weights_1gram = np.zeros(self.vocab_size, dtype=np.float64)
        self.weights_2gram = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        self.weights_context = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        self.is_trained = False

    def _softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        temp = max(1e-4, float(temperature))
        scaled = logits / temp
        shifted = scaled - np.max(scaled)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    def _get_logits(self, context_tokens: List[str]) -> np.ndarray:
        """Compute next-token logits given the context sequence."""
        logits = np.copy(self.weights_1gram)
        if not context_tokens:
            bos_id = self.token2id["<BOS>"]
            logits += self.weights_2gram[bos_id]
            return logits

        # Bigram transition from immediately preceding token
        last_tok = context_tokens[-1]
        last_id = self.token2id.get(last_tok, self.token2id["<UNK>"])
        logits += 1.8 * self.weights_2gram[last_id]

        # Multi-token context attention over preceding sequence
        for dist, tok in enumerate(reversed(context_tokens[:-1])):
            decay = 1.0 / (dist + 2.0)
            tok_id = self.token2id.get(tok, self.token2id["<UNK>"])
            logits += decay * self.weights_context[tok_id]

        return logits

    def continued_pretraining(
        self,
        train_sequences: List[List[str]],
        val_sequences: Optional[List[List[str]]] = None,
        steps: int = 100,
        learning_rate: float = 0.08,
        eval_every: int = 20,
    ) -> List[Dict[str, Any]]:
        """Perform continued pretraining updates to learn incident sequence transitions.

        Optimizes next-token cross-entropy loss over training incident sequences.
        """
        bos_id = self.token2id["<BOS>"]
        eos_id = self.token2id["<EOS>"]
        history: List[Dict[str, Any]] = []

        # Count frequencies for warm-start initialization
        unigram_counts = np.ones(self.vocab_size) * self.smoothing
        bigram_counts = np.ones((self.vocab_size, self.vocab_size)) * self.smoothing
        context_counts = np.ones((self.vocab_size, self.vocab_size)) * self.smoothing

        for seq in train_sequences:
            token_ids = [bos_id] + [self.token2id.get(t, self.token2id["<UNK>"]) for t in seq] + [eos_id]
            for i in range(1, len(token_ids)):
                target = token_ids[i]
                prev = token_ids[i-1]
                unigram_counts[target] += 1.0
                bigram_counts[prev, target] += 1.0
                for c in token_ids[max(0, i-self.context_window):i-1]:
                    context_counts[c, target] += 1.0

        # Convert initial counts to log-likelihood base weights
        self.weights_1gram = np.log(unigram_counts / np.sum(unigram_counts))
        self.weights_2gram = np.log(bigram_counts / np.sum(bigram_counts, axis=1, keepdims=True))
        self.weights_context = np.log(context_counts / np.sum(context_counts, axis=1, keepdims=True))

        # Iterative pretraining gradient update steps
        n_seqs = len(train_sequences)
        recent_losses: List[float] = []

        for step in range(1, steps + 1):
            seq_idx = self.rng.integers(0, n_seqs)
            seq = train_sequences[seq_idx]
            
            step_loss = 0.0
            n_tokens = 0
            
            # Causal autoregressive token loop
            for t in range(len(seq)):
                context = seq[:t]
                target_tok = seq[t]
                target_id = self.token2id.get(target_tok, self.token2id["<UNK>"])
                
                logits = self._get_logits(context)
                probs = self._softmax(logits)
                prob_target = max(1e-12, probs[target_id])
                
                loss = -math.log(prob_target)
                step_loss += loss
                n_tokens += 1
                
                # Gradient update (Cross-Entropy Gradient = probs - one_hot)
                grad = probs.copy()
                grad[target_id] -= 1.0
                
                # Update 1gram and 2gram weights
                self.weights_1gram -= (learning_rate * 0.1) * grad
                if context:
                    prev_id = self.token2id.get(context[-1], self.token2id["<UNK>"])
                    self.weights_2gram[prev_id] -= learning_rate * grad
                else:
                    self.weights_2gram[bos_id] -= learning_rate * grad

            avg_step_loss = step_loss / max(1, n_tokens)
            recent_losses.append(avg_step_loss)

            if step % eval_every == 0 or step == steps:
                train_loss = float(np.mean(recent_losses))
                train_ppl = float(math.exp(min(20.0, train_loss)))
                val_eval = self.evaluate(val_sequences) if val_sequences else None
                val_loss = val_eval["cross_entropy_loss"] if val_eval else None
                val_ppl = val_eval["perplexity"] if val_eval else None

                record = {
                    "step": step,
                    "train_loss": round(train_loss, 4),
                    "train_perplexity": round(train_ppl, 2),
                    "val_loss": round(val_loss, 4) if val_loss is not None else None,
                    "val_perplexity": round(val_ppl, 2) if val_ppl is not None else None,
                }
                history.append(record)
                recent_losses = []

        self.is_trained = True
        return history

    def predict_next(
        self,
        current_sequence: List[str],
        top_k: int = 3,
        temperature: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """Predict the most likely next incident types given current situation."""
        logits = self._get_logits(current_sequence)
        
        # Zero-out special tokens so we only predict real incident candidates
        for spec in self.special_tokens:
            logits[self.token2id[spec]] = -1e9

        probs = self._softmax(logits, temperature=temperature)
        
        # Rank by probability
        ranked_indices = np.argsort(-probs)[:top_k]
        results = [(self.id2token[idx], float(probs[idx])) for idx in ranked_indices]
        return results

    def evaluate(self, test_sequences: List[List[str]]) -> Dict[str, Any]:
        """Evaluate next-incident prediction quality across test sequences."""
        if not test_sequences:
            return {"error": "no test sequences provided"}

        total_loss = 0.0
        total_eval_tokens = 0
        top1_hits = 0
        top3_hits = 0
        reciprocal_ranks: List[float] = []

        for seq in test_sequences:
            for t in range(len(seq)):
                context = seq[:t]
                target = seq[t]
                target_id = self.token2id.get(target, self.token2id["<UNK>"])

                logits = self._get_logits(context)
                probs = self._softmax(logits)
                
                loss = -math.log(max(1e-12, probs[target_id]))
                total_loss += loss
                total_eval_tokens += 1

                # Exclude special tokens for ranking
                cand_probs = [(tok, probs[self.token2id[tok]]) for tok in self.vocab]
                cand_probs.sort(key=lambda x: -x[1])
                ranked_names = [tok for tok, _ in cand_probs]

                if target in ranked_names:
                    rank = ranked_names.index(target) + 1
                    reciprocal_ranks.append(1.0 / rank)
                    if rank == 1:
                        top1_hits += 1
                    if rank <= 3:
                        top3_hits += 1
                else:
                    reciprocal_ranks.append(0.0)

        avg_loss = total_loss / max(1, total_eval_tokens)
        ppl = math.exp(min(20.0, avg_loss))
        top1_acc = top1_hits / max(1, total_eval_tokens)
        top3_acc = top3_hits / max(1, total_eval_tokens)
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        return {
            "eval_tokens": total_eval_tokens,
            "cross_entropy_loss": round(avg_loss, 4),
            "perplexity": round(ppl, 2),
            "top1_accuracy": round(top1_acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "mrr": round(mrr, 4),
        }


def milestone_3(
    steps: int = 100,
    learning_rate: float = 0.08,
) -> Dict[str, Any]:
    """Milestone 3: Train an LM to predict the next likely incident given a current situation.

    Uses language model continued pretraining over incident sequences, calculating
    cross-entropy loss, perplexity, and Top-K ranking accuracy.
    """
    reports = get_default_incident_reports()
    sequences = [r.ordered_sequence for r in reports]

    # Split 80% train, 20% validation
    n_train = max(1, int(len(sequences) * 0.8))
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:]

    model = IncidentSequenceLM(vocab=INCIDENT_TYPES, seed=42)
    history = model.continued_pretraining(
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        steps=steps,
        learning_rate=learning_rate,
        eval_every=max(1, steps // 5),
    )

    eval_results = model.evaluate(val_seqs)

    # Test sample prediction
    sample_context = ["traffic_collision", "fuel_leak"]
    predictions = model.predict_next(sample_context, top_k=3)

    return {
        "milestone": 3,
        "status": "completed",
        "description": "Incident sequence language model trained via continued pretraining",
        "training_history": history,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_train_perplexity": history[-1]["train_perplexity"] if history else None,
        "evaluation_metrics": eval_results,
        "sample_context": sample_context,
        "sample_predictions": [{"incident": inc, "probability": round(prob, 4)} for inc, prob in predictions],
    }


# --- Milestone 4: Implement a simple text-based game where the player makes resource allocation decisions based on predicted events.
# Hint: Implement the game logic, allowing players to make decisions based on the LM's predictions.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Evaluate player performance based on efficient resource use and accurate predictions.
# Hint: Define and apply metrics to evaluate player performance based on resource use and prediction accuracy.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
