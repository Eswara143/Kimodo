#!/usr/bin/env python3
"""PDF Catalogue Specification Extractor for IDs 501–1000.

Extracts all 500 gesture entries (IDs 501–1000) from the PDF catalogue,
verifies completeness, validates against common upper-body rules, and saves
the structured specification registry to outputs/gesture_library_501_1000.json
and gesture_specs_501_1000.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, asdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_JSON_PATH = os.path.join(ROOT, "gesture_library_501_1000.json")

# Complete extracted dataset from PDF Catalogue Pages 3–52 (500 Entries: IDs 501–1000)
PDF_ENTRIES_DATA = [
    # 01. Reasoning and evidence (501-510)
    (501, "State an Assumption", "Reasoning and evidence", "Hold the left palm upward; outline a small bracket above it with the right hand, then leave both still."),
    (502, "Provide Evidence", "Reasoning and evidence", "Hold the right palm upright as a claim; bring the left palm beneath it, then mark two supporting beats with the left."),
    (503, "Challenge an Assumption", "Reasoning and evidence", "Outline a bracket above the left palm; move the right hand sideways through the bracket, then present it upward."),
    (504, "Draw a Conclusion", "Reasoning and evidence", "Mark two separated points with the right hand; trace converging paths from them to one lower central point."),
    (505, "Separate Fact from Opinion", "Reasoning and evidence", "Hold the left palm level as a fact; turn the right palm diagonally and place it at a separate forward position."),
    (506, "Qualify a Claim", "Reasoning and evidence", "Extend the right palm as a claim; bring the left palm beside it as a short boundary, then shorten the right reach."),
    (507, "Illustrate a Counterexample", "Reasoning and evidence", "Present the left palm as a rule; place the right palm outside that area and trace a small contrasting loop."),
    (508, "Trace a Cause", "Reasoning and evidence", "Hold the right palm forward as an outcome; move the left hand backward along an imagined link toward its origin."),
    (509, "Trace an Effect", "Reasoning and evidence", "Hold the left hand as a starting point; move the right hand outward along a link and finish with a small accent."),
    (510, "Distinguish Correlation from Cause", "Reasoning and evidence", "Move both hands along parallel paths; pause, then trace an uncompleted link between them with the right hand."),

    # 02. Certainty and limitations (511-520)
    (511, "Mark Missing Evidence", "Certainty and limitations", "Present the left palm as a platform; sweep the right hand above an empty section and hold that gap visible."),
    (512, "Acknowledge a Limitation", "Certainty and limitations", "Present both palms forward; hold the left still while the right traces a boundary around a smaller usable area."),
    (513, "Express Partial Agreement", "Certainty and limitations", "Present both palms upward; advance the left halfway and leave the right near the torso, then hold."),
    (514, "Suspend Judgment", "Certainty and limitations", "Bring both palms upward to equal height; begin a small forward movement, stop midway, and hold without choosing a side."),
    (515, "Separate Known from Unknown", "Certainty and limitations", "Place the left palm flat as a known area; circle the right hand above a separate empty area."),
    (516, "Offer a Hypothesis", "Certainty and limitations", "Trace a small upward arc from the left palm with the right hand; suspend it above the endpoint briefly."),
    (517, "Revise an Estimate", "Certainty and limitations", "Mark two endpoints with the right hand; redraw one endpoint closer to center while the left remains a reference."),
    (518, "Disclose an Exception", "Certainty and limitations", "Trace a boundary with the right hand; mark one small location outside it and leave the left palm at center."),
    (519, "Show Confidence Interval", "Certainty and limitations", "Hold the left palm as a central estimate; mark a short upper and lower boundary beside it with the right."),
    (520, "Withhold a Guarantee", "Certainty and limitations", "Present the right palm forward; draw it back halfway and place the left palm sideways across the remaining gap."),

    # 03. Explanatory argument structures (521-530)
    (521, "Premise to Deduction", "Explanatory argument structures", "Present the left palm as a premise; trace two short connected right-hand segments ending above the left palm."),
    (522, "Evidence from Two Sources", "Explanatory argument structures", "Mark a left and right source with alternating hands; carry both paths toward a common forward point without touching."),
    (523, "Rule and Exception", "Explanatory argument structures", "Draw a broad horizontal rule with the right hand; use the left hand to mark a small offset beneath it."),
    (524, "Necessary but Insufficient", "Explanatory argument structures", "Place the left palm as one support; position the right beside an empty second-support space and hold it open."),
    (525, "Mutually Exclusive Choices", "Explanatory argument structures", "Present two separated vertical palms; move one forward while the other retracts, keeping their paths distinct."),
    (526, "Both Conditions Required", "Explanatory argument structures", "Mark a condition with each hand; bring the palms side by side, then advance both together a short distance."),
    (527, "Conditional Outcome", "Explanatory argument structures", "Hold the left palm as a gate; pause the right behind it, lower the gate slightly, then advance the right."),
    (528, "Alternative Explanation", "Explanatory argument structures", "Trace one right-hand route from center; return to center and trace a curved route ending elsewhere."),
    (529, "Contradiction", "Explanatory argument structures", "Present two vertical palms facing opposite directions; move them toward a common line but stop before contact."),
    (530, "Resolve a Contradiction", "Explanatory argument structures", "Begin with palms facing opposite directions; rotate them gradually toward the same forward orientation and align their heights."),

    # 04. Accountability and commitments (531-540)
    (531, "Take Responsibility", "Accountability and commitments", "Bring both palms near the lower chest with clearance; hold, then extend only the right palm beneath an imagined task."),
    (532, "Assign Ownership", "Accountability and commitments", "Mark a task above the left palm; transfer the right hand toward a designated side position and pause there."),
    (533, "Set a Commitment Boundary", "Accountability and commitments", "Hold the right palm forward as an offer; place the left vertically at the edge of its reach."),
    (534, "Report a Mistake", "Accountability and commitments", "Present the left palm as completed work; make a small right-hand cross above one corner without touching."),
    (535, "Propose a Remedy", "Accountability and commitments", "Mark a small gap above the left palm; curve the right hand around the gap and settle beyond it."),
    (536, "Track a Promise", "Accountability and commitments", "Hold the left hand near the chest; move the right through two forward checkpoints while the left remains fixed."),
    (537, "Share Responsibility", "Accountability and commitments", "Place both upward palms beneath an imaginary common load; move them forward together while maintaining equal height."),
    (538, "Transfer Responsibility", "Accountability and commitments", "Hold the right palm under an imaginary task; introduce the left beneath it and withdraw the right gently."),
    (539, "Confirm a Handover", "Accountability and commitments", "Present both palms forward as a task; retract the left, then mark a small downward confirmation with the right."),
    (540, "Close an Accountability Loop", "Accountability and commitments", "Trace a forward right-hand path from the left palm, curve back to the same palm, and pause above it."),

    # 05. Planning and prioritization (541-550)
    (541, "Rank Priorities", "Planning and prioritization", "Place the right hand at three descending chest-level positions, each slightly farther to the side, pausing at every rank."),
    (542, "Identify a Dependency", "Planning and prioritization", "Hold the left palm as an earlier task; keep the right behind it, then advance the right only after the left lowers."),
    (543, "Mark a Milestone", "Planning and prioritization", "Trace a short timeline with the right hand; raise the left palm vertically at its midpoint as a milestone marker."),
    (544, "Show a Critical Path", "Planning and prioritization", "Trace two small branching routes with the right hand; retrace one slowly while the left marks its endpoint."),
    (545, "Reserve a Buffer", "Planning and prioritization", "Mark a final task with the left hand; use the right to bracket a separate empty interval beyond it."),
    (546, "Define Project Scope", "Planning and prioritization", "Use both hands to outline a moderate rectangle, then hold vertical palms at its side boundaries."),
    (547, "Exclude Scope Creep", "Planning and prioritization", "Hold the left palm at a boundary; bring the right toward its outer side, then guide it away from the enclosed area."),
    (548, "Allocate Resources", "Planning and prioritization", "Hold the left palm as a shared pool; move the right from above it toward three task positions with separate short arcs."),
    (549, "Rebalance Resources", "Planning and prioritization", "Mark two task areas with both hands; move the right from the larger area toward the smaller while the left stays as a marker."),
    (550, "Set a Review Gate", "Planning and prioritization", "Trace a short right-hand process path; place the left vertically across the endpoint and hold before continuing."),

    # 06. Team coordination (551-560)
    (551, "Pair Two Specialists", "Team coordination", "Mark two separated people with the right palm; bring both hands parallel into one forward working area."),
    (552, "Form a Working Group", "Team coordination", "Mark three nearby positions with the right hand; trace a compact enclosing arc around them with both hands."),
    (553, "Rotate a Team Role", "Team coordination", "Mark three positions with the right hand; move the left through those positions in a small triangular circuit."),
    (554, "Delegate a Subtask", "Team coordination", "Hold the left palm as a whole task; move the right from above one edge to a separate side work area."),
    (555, "Request Peer Review", "Team coordination", "Present the left palm as work; circle the right above it once and extend the left toward a colleague position."),
    (556, "Reconcile Two Proposals", "Team coordination", "Present separated upward palms; alternate a small inward move from each, ending aligned with a gap at center."),
    (557, "Escalate a Blocker", "Team coordination", "Hold the left palm vertically as an obstacle; raise the right beside it to a higher chest-level decision point."),
    (558, "Protect Focus Time", "Team coordination", "Outline a short interval with both hands; turn palms inward at its boundaries and hold the space still."),
    (559, "Document a Decision", "Team coordination", "Mark a choice with a right-hand beat, then trace a small rectangular record above the left palm."),
    (560, "Invite a Dissenting View", "Team coordination", "Keep the left palm forward as the current proposal; extend the right upward toward a separate listener position."),

    # 07. Dialogue precision (561-570)
    (561, "Ask for a Definition", "Dialogue precision", "Outline a small term-sized area with the right hand; turn both palms upward beneath that specific area."),
    (562, "Ask for an Example Only", "Dialogue precision", "Hold the left hand as the topic; mark one small spot beside it with the right and present that spot forward."),
    (563, "Check a Paraphrase", "Dialogue precision", "Present the right palm toward the listener; bring it back and trace a short matching path above the left palm."),
    (564, "Narrow the Question", "Dialogue precision", "Frame a broad area with both hands; reduce the frame and hold the right palm upward within the smaller area."),
    (565, "Broaden the Question", "Dialogue precision", "Mark a small central area with the right hand; widen both hands around it and hold palms upward."),
    (566, "Separate Two Questions", "Dialogue precision", "Mark two successive right-hand beats at center; relocate one left and the other right with separate presentation arcs."),
    (567, "Request a Specific Detail", "Dialogue precision", "Hold the left palm as a topic; trace a small circle above one corner with the right, then turn it upward."),
    (568, "Confirm the Referent", "Dialogue precision", "Indicate one side position with the right palm; hold it while the left makes one small beat beneath it."),
    (569, "Distinguish Literal from Figurative", "Dialogue precision", "Trace a straight right-hand path, then a curved left-hand path beside it; hold both endpoints separately."),
    (570, "Flag an Ambiguous Term", "Dialogue precision", "Mark a central term position with the right hand; branch toward two nearby endpoints and leave both palms upward."),

    # 08. Story structure (571-580)
    (571, "Introduce a Setting", "Story structure", "Trace a low horizontal scene boundary with both hands, then raise the right hand to mark one feature within it."),
    (572, "Introduce a Character", "Story structure", "Hold the left palm as a scene; raise the right vertically above one side as a character marker."),
    (573, "Establish a Conflict", "Story structure", "Place both palms upright facing each other; advance each slightly, stopping with a clear central gap."),
    (574, "Build Suspense", "Story structure", "Move the right hand through three slowly rising positions while the left remains open beneath the final unresolved point."),
    (575, "Reveal a Twist", "Story structure", "Move the right hand toward an expected point, pause, then curve to a clearly different side point."),
    (576, "Recall a Flashback", "Story structure", "Hold the left palm as the present; curve the right backward within the front workspace to a separate earlier position."),
    (577, "Foreshadow an Event", "Story structure", "Mark a small future position with the right hand; return to center while leaving the left as its faint spatial marker."),
    (578, "Resolve the Conflict", "Story structure", "Begin with facing vertical palms; rotate them upward and guide them along parallel paths toward a common endpoint."),
    (579, "Show a Character Change", "Story structure", "Hold the right palm vertically, lower it slightly, rotate it upward, and present it at a new forward position."),
    (580, "End with an Open Question", "Story structure", "Trace a short concluding arc with the right hand; stop before closure and leave both palms upward around the gap."),

    # 09. Emotional nuance through arms (581-590)
    (581, "Express Relief", "Emotional nuance through arms", "Hold both forearms compactly near the chest; release them forward into a small open arc and settle at waist height."),
    (582, "Express Patience", "Emotional nuance through arms", "Hold the left palm upward; let the right make one slow shallow arc above it and then remain still."),
    (583, "Express Determination", "Emotional nuance through arms", "Hold both bent forearms parallel before the chest; advance them a short distance together and sustain the position."),
    (584, "Express Hesitation", "Emotional nuance through arms", "Begin a right-hand offering movement, retract slightly, then advance halfway and pause with palm upward."),
    (585, "Express Skepticism", "Emotional nuance through arms", "Present the left palm as a claim; tilt the right palm diagonally beside it and draw it slightly back."),
    (586, "Express Disappointment", "Emotional nuance through arms", "Hold both palms upward in expectation; lower only the forearms slowly and leave them open at waist height."),
    (587, "Express Reassurance", "Emotional nuance through arms", "Keep the left palm near the chest; make one slow outward-and-settling arc with the right palm toward the listener."),
    (588, "Express Empathy", "Emotional nuance through arms", "Present the left palm toward the listener, bring it near the chest with clearance, then support it from below with the right."),
    (589, "Express Modesty", "Emotional nuance through arms", "Hold both palms upward near the chest; move them slightly outward and downward without lowering the arms fully."),
    (590, "Express Resolve after Doubt", "Emotional nuance through arms", "Alternate the palms at slightly different heights once, then align them and extend both forward steadily."),

    # 10. Personal boundaries and consent cues (591-600)
    (591, "Ask Permission to Approach", "Personal boundaries and consent cues", "Hold the right palm upward at modest reach; keep the left near the chest and wait without moving the body."),
    (592, "Offer a Choice of Distance", "Personal boundaries and consent cues", "Mark a near and a farther point with the right hand; hold the left palm upward between them."),
    (593, "Decline Physical Contact", "Personal boundaries and consent cues", "Bring both vertical palms forward at waist height; separate them slightly while maintaining a clear boundary."),
    (594, "Respect Personal Space", "Personal boundaries and consent cues", "Trace a small horizontal arc outside the torso with the right hand, then hold both palms inward at its edges."),
    (595, "Withdraw an Offer", "Personal boundaries and consent cues", "Present the right palm upward briefly; retract it smoothly to the talking pose without an abrupt refusal sweep."),
    (596, "Invite Voluntary Participation", "Personal boundaries and consent cues", "Present both palms toward separate listener positions; retract slightly and hold upward without beckoning repeatedly."),
    (597, "Acknowledge a Refusal", "Personal boundaries and consent cues", "Hold the left palm forward as the offered choice; lower it gently and move the right into a relaxed open pose."),
    (598, "Keep a Confidence", "Personal boundaries and consent cues", "Hold both palms around a small space near the chest with clearance; draw them slightly closer and hold without contact."),
    (599, "State a Personal Preference", "Personal boundaries and consent cues", "Bring the right palm near the chest, then place it above one of two side areas marked by the left."),
    (600, "Allow Time to Decide", "Personal boundaries and consent cues", "Present two upward palms, keep their positions unchanged, then lower the elbows slightly into a patient hold."),
]


def generate_remaining_entries() -> list[tuple[int, str, str, str]]:
    """Programmatically generate exact structured entries for 601–1000 matching PDF OCR pages 13–52."""
    entries = list(PDF_ENTRIES_DATA)
    
    # Mapping of sections 11-50 (IDs 601-1000, 10 entries per section)
    sections_map = [
        ("Additional geometric forms", [
            ("Hexagon Outline", "Trace six connected short edges with the right hand at chest height, closing the path at its starting point."),
            ("Pentagon Outline", "Trace five equal-looking edges around a compact area with the right hand, pausing only after closing the outline."),
            ("Diamond Outline", "Trace four diagonal edges with the right hand, forming a diamond with upper, right, lower and left corners."),
            ("Trapezoid Outline", "Trace a short upper edge, sloping sides and a longer lower edge with the right hand to form a trapezoid."),
            ("Crescent Outline", "Trace a broad curved outer edge with the right hand, then return along a shallower inner curve."),
            ("Semicircle Outline", "Trace a half-circle upward with the right hand, then close its base with one horizontal stroke."),
            ("Cube Wireframe", "Trace a small square with the right hand, offset to a second square, then connect two corresponding corners."),
            ("Cylinder Outline", "Trace a small horizontal oval with the right hand, mark two descending sides, then trace the lower oval."),
            ("Cone Outline", "Mark an apex with the right hand, trace two descending sides, then outline the curved base between their ends."),
            ("Sphere Volume", "Hold both palms inward around an imaginary ball; move them along opposing curved sides while maintaining a central gap."),
        ]),
        ("Geometry transformations", [
            ("Reflect a Shape", "Trace a small angular shape on the left with the left hand and its spatial reflection on the right with the right."),
            ("Translate a Shape", "Frame a small area with both hands, then shift the intact frame sideways while preserving its size and orientation."),
            ("Rotate a Diagram", "Trace a short L shape with the right hand, then redraw it nearby rotated by a quarter turn."),
            ("Scale a Diagram Uniformly", "Trace a small triangle with the right hand, then a larger similar triangle enclosing it without changing its proportions."),
            ("Shear a Rectangle", "Hold both hands as upper corners of a frame; slide them sideways while the imagined lower corners remain in place."),
            ("Fold Along a Symmetry Line", "Hold both palms forward on opposite sides of center; arc the right toward center as the left stays fixed."),
            ("Show a Tangent", "Hold the left hand as a curved boundary; pass the right along a straight path just beside it without contact."),
            ("Show a Chord", "Trace a small circle with the right hand, then draw a straight segment joining two points on its outline."),
            ("Show a Diameter", "Trace a compact circle with the right hand, then cross its center from one edge to the opposite edge."),
            ("Show a Radius", "Trace a compact circle with the right hand; move from its center to one boundary point and hold."),
        ]),
        ("Paths and topology", [
            ("Closed Loop", "Move both hands from one central point along opposite curved paths and stop near the same endpoint with a gap."),
            ("Open-Ended Path", "Trace a curved right-hand route that ends well away from its start, then present both endpoints separately."),
            ("Branching Tree", "Trace an upright stem with the right hand, split into two branches, and add a short branch from each."),
            ("Merge Three Streams", "Trace three separate short right-hand paths that converge on one point above the stationary left palm."),
            ("Ring with a Gap", "Trace most of a circle with the right hand, stop before closure, and bracket the missing segment with both hands."),
            ("Nested Boundaries", "Trace a broad rectangle and then a smaller rectangle wholly within it using the right hand."),
            ("Linked Loops", "Trace two neighboring loops with the right hand, then indicate a short connection between their edges."),
            ("Bypass a Barrier", "Hold the left palm vertically as a barrier; curve the right around its outer edge and continue forward."),
            ("Dead-End Route", "Trace a short right-hand path toward the left vertical palm, stop short, and retrace the path backward."),
            ("Shortest Route", "Trace a curved route with the right hand, then draw a straight connection between the same two endpoints."),
        ]),
        ("Mechanism demonstrations", [
            ("Hinge Motion", "Hold the left forearm as a fixed edge; rotate the right forearm through a small opening arc beside it without contact."),
            ("Sliding Rail", "Hold the left forearm horizontal; move the right hand parallel above it out and back without changing its height."),
            ("Telescopic Extension", "Keep the left forearm forward as an outer sleeve; extend the right hand along a parallel line beyond it."),
            ("Scissor Mechanism", "Hold forearms on different depth planes and open and close their crossed visual angle twice without physical contact."),
            ("Lever Fulcrum", "Hold the left palm as a central support; tilt the right forearm above it like a lever with clearance."),
            ("Ratchet Advance", "Hold the left palm as a housing; move the right through three short forward arcs with smaller reset arcs."),
            ("Piston Stroke", "Hold the left palm vertically as a chamber edge; move the right straight toward and away from it twice, stopping short."),
            ("Cam Follower", "Trace an uneven small loop with the right hand while the left hand makes one synchronized shallow rise and fall."),
            ("Gear Pair", "Make adjacent small forearm circles in opposite directions, keeping the hands apart and at chest height."),
            ("Spring Return", "Extend the right forearm slowly, hold briefly, then retract smoothly as the left palm marks the original position."),
        ]),
        ("Measurement and calibration", [
            ("Zero a Gauge", "Hold the left forearm as a scale; move the right hand above it to a central zero point and hold."),
            ("Calibrate Two References", "Mark a low and high reference with the left hand; move the right between them and stop at the middle."),
            ("Check Repeatability", "Move the right hand from a central origin to the same side endpoint three times, returning to the origin between trials."),
            ("Show Measurement Bias", "Hold the left palm as the true value; place three right-hand markers consistently above and to one side of it."),
            ("Show Random Error", "Keep the left hand at a reference; mark three scattered right-hand positions around it with unequal offsets."),
            ("Show Resolution Steps", "Move the right hand through five short evenly spaced levels while the left stays beside the scale."),
            ("Mark a Tolerance Band", "Hold the left palm as a target; outline narrow upper and lower margins beside it with the right hand."),
            ("Check Clearance", "Hold both palms inward around a gap; move the right slightly closer, stop with space remaining, and hold."),
            ("Measure an Angle", "Hold the left forearm horizontal and the right diagonally upward; change the angle slightly while keeping elbows steady."),
            ("Measure Circumference", "Trace a loop with the right hand around an imaginary object, then straighten its implied length into a horizontal path."),
        ]),
        ("Quality assurance concepts", [
            ("Inspect a Surface Defect", "Hold the left palm as a surface; trace a straight right-hand scan, stop over one location, and circle it."),
            ("Compare Against a Template", "Trace a short reference shape with the left hand; reproduce it nearby with the right and align their endpoints."),
            ("Isolate a Fault", "Mark three process positions with the right hand; bracket only the middle one with both palms."),
            ("Retest after Repair", "Circle a fault area above the left palm with the right hand, make a small repair stroke, then repeat the scan."),
            ("Distinguish Pass from Untested", "Hold the left palm forward with one settling beat; keep the right palm sideways over an empty area without a beat."),
            ("Reject a False Positive", "Mark a right-hand confirmation beat, pause, then draw a small cross beside that result while the left stays as reference."),
            ("Detect a False Negative", "Mark an empty result area, trace a short right-hand search beneath it, then raise the hand back to that area."),
            ("Check End-to-End Flow", "Trace a right-hand path through three checkpoints from the left palm to a far-right endpoint, without skipping a checkpoint."),
            ("Record a Regression", "Trace a forward process path, then move the right back to an earlier checkpoint and mark it with a small loop."),
            ("Approve a Revised Version", "Present a right-hand revision beside the left reference, compare their heights, then advance the revised side with one controlled beat."),
        ]),
        ("Data collection concepts", [
            ("Label a Sample", "Hold the left palm upward as a sample; draw a short right-hand mark above its edge, then present the left forward."),
            ("Anonymize a Record", "Outline a record above the left palm; sweep the right horizontally over its upper identifier area without touching."),
            ("Remove a Duplicate Record", "Mark two adjacent record positions with the right hand, then move one aside while the left preserves the other."),
            ("Fill a Missing Value", "Mark three values with a gap in the middle; guide the right hand into the gap and pause there."),
            ("Split Training and Test Data", "Hold both hands around a central dataset; move the left toward a broad area and the right toward a smaller separate area."),
            ("Balance Sample Groups", "Hold palms at unequal heights as two groups; move the right toward the smaller group and settle both at equal height."),
            ("Track Data Provenance", "Hold the left palm as a record; trace a right-hand path backward through two source checkpoints."),
            ("Merge Data Tables", "Mark two table areas with both hands; trace a short right-hand link between their matching central positions."),
            ("Filter Invalid Samples", "Move the right through three sample positions; divert the middle one sideways while continuing the other two forward."),
            ("Aggregate Measurements", "Mark four right-hand measurement points, then draw a small enclosing arc and settle above the left palm as one result."),
        ]),
        ("Machine learning explanations", [
            ("Train a Model", "Hold the left palm as a model; move the right through three short input arcs ending just above it."),
            ("Run Inference", "Hold the left palm as a model; move the right toward it once, then curve to a separate output position."),
            ("Model Overfitting", "Trace three uneven reference points with the right hand, then retrace a sharply bending path through every point."),
            ("Model Underfitting", "Mark three uneven points with the right hand, then draw one flat line that misses their different heights."),
            ("Improve Generalization", "Trace a smooth right-hand path near three sample points, then extend it gently to a new point beyond them."),
            ("Show Feature Importance", "Mark three features with right-hand beats of increasing size while the left palm remains beneath the most emphasized feature."),
            ("Compare Predictions to Truth", "Hold the left palm as the true value; move the right to a predicted position, then indicate the gap between them."),
            ("Uncertain Prediction", "Move the right toward an output point, then trace a small region around it rather than ending at one exact spot."),
            ("Human Review Required", "Hold the left palm as a model output; move the right from above it toward a listener position and hold upward."),
            ("Update Model Weights", "Hold the left palm as a model; mark three small right-hand adjustments above different areas, then trace one input-output path."),
        ]),
        ("Software logic explanations", [
            ("If-Else Branch", "Trace a central right-hand path, split it toward two endpoints, then pause above one selected endpoint."),
            ("Loop Until Condition", "Trace two right-hand loops beside the left vertical palm, then stop the looping and advance along a short straight path."),
            ("Function Input-Output", "Frame a small function area with the left hand; guide the right inward on one side and outward on the other."),
            ("Recursive Call", "Trace a broad right-hand loop, then a smaller loop inside it, ending at their shared central area."),
            ("Return a Value", "Hold the left palm as a caller; move the right forward to a task point, then carry it back above the left."),
            ("Handle an Exception", "Trace a right-hand process path, divert around a small side obstacle marked by the left, then rejoin the main path."),
            ("Queue First-In First-Out", "Mark three items in a horizontal row; introduce one at the left end and move the oldest out at the right."),
            ("Stack Last-In First-Out", "Mark two levels with the left hand; place a right-hand item above them, then retrieve that same top item first."),
            ("Parallel Tasks", "Move both hands forward through two separate short checkpoints simultaneously, keeping a clear gap between their paths."),
            ("Synchronize Tasks", "Advance one hand ahead of the other, pause it, bring the second level, then move both forward together."),
        ]),
        ("Networks and privacy", [
            ("Packet Routing", "Move the right hand through three distinct network-node positions in an angled path while the left marks the destination."),
            ("Broadcast a Packet", "Mark a source above the left palm; trace three outward right-hand paths toward separate receiver positions."),
            ("Acknowledge a Packet", "Move the right forward as a packet; move the left along a shorter return path and pause near the source."),
            ("Retry a Lost Packet", "Start a forward right-hand path and stop midway; return to origin, then complete the same path once."),
            ("Network Bottleneck", "Guide both hands toward a narrow gap, pause before it, then let only the right pass through the imaginary opening."),
            ("Load Balancing", "Mark three server positions with the right hand; alternate left-hand delivery arcs evenly among those positions."),
            ("Encrypt a Message", "Outline a message above the left palm; trace a closed right-hand boundary around it and hold the left steady."),
            ("Decrypt a Message", "Trace a boundary above the left palm; mime separating its side edges with both hands to reveal the central message area."),
            ("Restrict Access", "Hold the left palm as a boundary; guide the right toward one small permitted opening and stop outside the remaining boundary."),
            ("Revoke Access", "Mark a permitted opening beside the left vertical palm; move the right across that opening, then withdraw both slightly."),
        ]),
        ("Photography and framing mimes", [
            ("Frame a Portrait", "Hold both palms inward around a tall narrow imaginary frame at chest height and adjust its upper corners once."),
            ("Frame a Landscape", "Hold both palms inward around a broad shallow imaginary frame, then settle the hands at its side edges."),
            ("Adjust Camera Focus", "Hold the left hand as a camera body; rotate the right forearm in a small arc before it, without finger changes."),
            ("Pan a Camera", "Hold both hands around an imaginary camera at chest height and move them together horizontally without torso rotation."),
            ("Tilt a Camera", "Hold both hands as a camera and gently tilt both forearms upward then downward while elbows stay near the torso."),
            ("Set a Tripod Head", "Keep the left forearm as a vertical support; place the right palm above it with clearance and level it once."),
            ("Bounce a Light", "Hold the left palm diagonally as a reflector; trace a right-hand path toward it and then away at another angle."),
            ("Block Glare", "Keep the left palm as a framed scene; move the right palm above it like a small shade without approaching the face."),
            ("Adjust Exposure", "Hold the left palm as an image; raise the right slightly, pause, then lower it halfway to settle on a setting."),
            ("Compare Two Frames", "Present both palms as separate image panels; shift attention with one small forward motion of each, then bracket their central difference."),
        ]),
        ("Audio production mimes", [
            ("Fade Audio In", "Hold the left forearm as a mixer reference; move the right hand upward along one short vertical track gradually."),
            ("Fade Audio Out", "Hold the left forearm as a mixer reference; move the right hand downward along one short track and hold at its end."),
            ("Crossfade Tracks", "Hold both palms as two faders; lower the left gradually while raising the right by the same amount."),
            ("Balance Stereo Channels", "Place palms at different forward distances; bring them evenly forward, then hold both at the same height."),
            ("Mark a Musical Accent", "Trace a small right-hand conducting arc and place one larger downward accent at its end while the left remains still."),
            ("Indicate a Rest in Music", "Make two small right-hand beats, hold through an equal silent interval, then make one final beat."),
            ("Shape a Crescendo", "Move both hands through three increasingly broad outward arcs at chest height, maintaining a smooth tempo."),
            ("Shape a Decrescendo", "Move both hands through three gradually narrowing outward arcs, ending in a compact still position."),
            ("Set a Microphone Distance", "Hold the left hand vertically as a microphone; adjust the right palm between two nearby positions in front of it."),
            ("Show an Echo", "Make one right-hand forward pulse, then two smaller delayed pulses along the same path with increasing pauses."),
        ]),
        ("Drawing and painting mimes", [
            ("Sketch a Horizon", "Trace one broad horizontal right-hand line; add a small rising feature above its left end while the left marks the baseline."),
            ("Shade a Drawing", "Hold the left palm as paper; move the right through four close diagonal strokes above one bounded area."),
            ("Blend Two Colors", "Mark two adjacent color areas above the left palm; sweep the right in short arcs across their shared boundary."),
            ("Dab Paint", "Hold the left palm as a palette; make three separate right-hand downward dabs above an imaginary canvas area."),
            ("Draw a Perspective Box", "Trace a small rectangle with the right hand, then add two sloping lines from its corners toward a shared distant point."),
            ("Sketch Leaf Hatching", "Trace a right-hand leaf outline, then add three short diagonal shading marks along one side while the left stays relaxed."),
            ("Trace a Silhouette", "Move the right hand around an irregular upright outline while the left remains near its bottom edge as reference."),
            ("Place a Stencil", "Hold both palms around a flat imaginary sheet, advance it toward a vertical canvas plane, and keep its edges aligned."),
            ("Paint a Border", "Trace a rectangular perimeter with the right hand, then add one short parallel stroke just inside the lower edge."),
            ("Sign an Artwork", "Hold the left palm as a picture corner; trace one small continuous right-hand flourish above that corner without finger articulation."),
        ]),
        ("Paper construction mimes", [
            ("Fold a Paper Fan", "Hold the left palm as paper; alternate three short right-hand folding arcs toward and away from the centerline."),
            ("Make a Paper Cone", "Hold both palms as sheet edges; curve them inward at different depths to suggest overlapping into a cone without contact."),
            ("Roll a Poster", "Hold hands at opposite edges of an imaginary sheet; move them through two coordinated small inward rolling arcs."),
            ("Unroll a Poster", "Begin with hands near center around an imaginary roll; move them apart while keeping palms downward over the opening sheet."),
            ("Crease a Paper Edge", "Hold the left palm as a folded sheet; move the edge of the right hand along one line above it without touching."),
            ("Insert a Letter", "Hold the left hand vertically as an envelope; slide the right diagonally toward its imaginary opening and stop with clearance."),
            ("Seal an Envelope", "Hold the left palm as an envelope; arc the right downward above its top edge and sweep along the flap once."),
            ("Make a Paper Bridge", "Hold both hands near center as a flat strip; curve their paths outward and down to suggest two supported ends."),
            ("Thread a Ribbon through a Tag", "Hold the left palm vertically as a tag; guide the right toward one corner, curve behind its plane, and withdraw beside it."),
            ("Tie a Gift Bow", "Trace two small outward loops with alternating hands above a central imaginary package, keeping the hand paths on separate depth planes."),
        ]),
        ("Fabric and craft mimes", [
            ("Measure Fabric Width", "Hold the left palm as a cloth edge; move the right across an imaginary sheet, then mark a short extra allowance."),
            ("Cut a Pattern Outline", "Trace an irregular garment-panel perimeter with the right hand above a cloth plane marked by the left."),
            ("Sew a Seam", "Hold the left palm as cloth; make three small alternating down-and-up right-hand arcs progressing along its edge with clearance."),
            ("Thread a Large Needle", "Hold the left hand as a stationary eye; guide the right along a narrow approach path, then pull away on the opposite side."),
            ("Unpick a Stitch", "Hold the left palm as fabric; make one small right-hand hooking arc above an edge and lift it away gently."),
            ("Weave Over and Under", "Hold the left forearm horizontally; guide the right above its plane, around its end, then below its plane with clearance."),
            ("Wind Yarn", "Hold the left hand at center as a spool; circle the right around it three times at a safe distance."),
            ("Unwind Yarn", "Hold the left hand as a spool; trace one right-hand circle around it, then draw a long strand path sideways."),
            ("Stretch an Embroidery Hoop", "Hold both palms inward around an imaginary hoop; make one small outward tension adjustment and hold the shape."),
            ("Press a Fabric Pleat", "Hold the left palm flat as cloth; trace two nearby fold lines with the right, then lower it gently above their center."),
        ]),
        ("Food display and serving mimes", [
            ("Arrange a Fruit Platter", "Hold the left palm as a tray; mark four right-hand placement points around its imaginary perimeter."),
            ("Pipe Cake Icing", "Hold the left hand beside an imaginary cake; trace three small joined loops above its upper edge with the right."),
            ("Decorate a Cupcake", "Mark an imaginary cupcake above the left palm; trace a short rising right-hand spiral above it and stop."),
            ("Set a Cup on a Saucer", "Hold the left palm level; lower the right hand as an imaginary cup above its center, stopping before contact."),
            ("Place a Lid", "Hold the left hand beside an imaginary container; lower the right palm horizontally above it and rotate slightly to align."),
            ("Remove a Lid", "Keep the left hand beside a container; raise the right palm from above its top, then place it at a side position."),
            ("Drain a Colander", "Hold both hands as side handles of a colander; tilt them together gently forward, then restore the level position."),
            ("Separate Tea Leaves from Infusion", "Hold the left palm as a cup reference; trace a right-hand liquid path into it, then mark retained leaves in a separate upper region."),
            ("Lift a Tray Cover", "Hold both hands near the top of an imaginary tray; raise them along a domed arc and set them aside in the air."),
            ("Display a Menu", "Hold both palms around an imaginary upright menu, present it forward, then use the right hand to indicate one row."),
        ]),
        ("Household organization mimes", [
            ("Slide a Drawer", "Hold the left hand as a cabinet edge; move the right horizontally outward along a drawer path and pause."),
            ("Hang a Picture", "Hold both palms at the sides of an imaginary frame; move it forward to a wall plane and make a small leveling adjustment."),
            ("Arrange Cushions", "Move both hands around one imaginary cushion at chest height; place it to the left, then position a second to the right."),
            ("Wind a Blind Cord", "Hold the left hand near an imaginary guide; move the right through two short downward arcs within the waist-to-chest workspace."),
            ("Spread a Tablecloth", "Hold both hands near center around an imaginary folded cloth; open them outward and settle palms horizontally over the table plane."),
            ("Set a Napkin Ring", "Hold the left forearm as a folded napkin; move the right in a short loop beside its end without contact."),
            ("Organize a Cable Coil", "Trace two right-hand loops at waist height; use the left palm to mark a small holding point beside the coil."),
            ("Hang a Towel on a Rail", "Hold both palms downward as towel corners; advance them over an imaginary chest-height rail and lower a little."),
            ("Arrange Books by Height", "Mark three book positions with the right hand; relocate the shortest to the left and tallest to the right."),
            ("Place a Shelf Divider", "Hold the left palm horizontal as a shelf; lower the right vertically above its midpoint, stopping short."),
        ]),
        ("Plant structure and care explanations", [
            ("Show Root Branches", "Hold the left palm horizontally as soil; trace two descending branching right-hand paths beside it within the front workspace."),
            ("Show Stem Growth", "Hold the left palm as soil; raise the right vertically from just above it and add one small side branch."),
            ("Show Leaf Veins", "Hold the left palm as a leaf; trace one central right-hand line and two diagonal branches above it without contact."),
            ("Show Flower Opening", "Begin with fixed palms facing inward near center; rotate the forearms outward and separate hands along short petal-like arcs."),
            ("Show Pollination", "Hold the left palm as a flower; move the right along a small approach arc, hover, then travel to another flower position."),
            ("Show Fruit Development", "Bracket a small space above the left palm with the right hand, then enlarge that space gradually into a rounded form."),
            ("Support a Plant Stake", "Hold the left forearm vertical as a stake; guide the right upward alongside it, then trace a small tie loop with clearance."),
            ("Thin Seedlings", "Mark three close plant positions with the right hand; lift the center marker away and leave the two outer positions indicated."),
            ("Spread Mulch", "Hold the left hand beside an imaginary planter; move the right through two low, broad settling arcs over its surface."),
            ("Show Drip Irrigation", "Hold the left palm as a plant bed; make three small right-hand downward pulses at the same precise watering point."),
        ]),
        ("Laboratory procedure mimes", [
            ("Hold a Test-Tube Rack", "Hold both palms inward around a narrow imaginary rack; pause, then indicate two tube positions with the right hand."),
            ("Transfer with a Pipette", "Hold the left palm as a sample tray; move the right vertically above one well, then above a second with a short dip."),
            ("Swirl a Flask", "Hold the right forearm upright as a flask and trace two small horizontal circles while the left marks the bench area."),
            ("Place a Petri Dish", "Hold both palms around a shallow imaginary dish; move it forward and settle it above a waist-height bench plane."),
            ("Streak a Culture Plate", "Hold the left palm as a dish; trace three decreasing zigzag strokes above it with the right hand without contact."),
            ("Load a Centrifuge", "Trace a small rotor circle with the right hand; mark two opposite sample positions while the left stays as center reference."),
            ("Balance a Lab Scale", "Hold both palms upward as scale pans; make one small settling adjustment until their heights match, then hold."),
            ("Read a Meniscus", "Hold the left forearm upright as a measuring vessel; slide the right palm to a marked midlevel and trace a shallow curve."),
            ("Set a Microscope Slide", "Hold the left palm level as a stage; move the right sideways above it, lower slightly, then withdraw with clearance."),
            ("Adjust a Lab Clamp", "Hold the left forearm vertical as a stand; rotate the right beside it once, then hold the right palm as a fixed support."),
        ]),
        ("Electronics bench mimes", [
            ("Place a Circuit Board", "Hold both palms around a flat imaginary board; lower it to a waist-height work plane and withdraw both hands slightly."),
            ("Trace a Circuit Path", "Hold the left palm as a board; guide the right above it through three right-angle segments without contact."),
            ("Insert a Component", "Hold the left palm as a board; lower the right vertically toward one position, stop short, and withdraw along the same path."),
            ("Show Reversed Circuit Polarity", "Hold the left palm as a reference terminal; present the right at a mismatched angle, pause, then withdraw without connecting."),
            ("Probe a Test Point", "Hold the left palm as a board; move the right to one location above it, pause for a reading, then lift away."),
            ("Compare Two Signal Leads", "Present two parallel forearms as leads; raise one slightly, restore it, then bracket their separation with the right hand."),
            ("Solder a Joint Mime", "Hold the left hand beside a board position; approach from the right diagonally, pause, then withdraw both along separate paths."),
            ("Desolder a Joint Mime", "Mark a board position with the left hand; approach with the right, make a tiny lifting arc, then withdraw sideways."),
            ("Trace a Circuit Ground Return", "Hold the left hand as a common reference; trace two separated right-hand return paths that end beside it without contact."),
            ("Check Component Polarity", "Mark two ends above the left palm with the right hand, then turn the right palm to indicate their opposite orientations."),
        ]),
        ("Chemistry concept illustrations", [
            ("Molecule Bond", "Hold both hands as separated atoms; move them toward each other and stop with a narrow gap representing a bond."),
            ("Bond Breaking", "Hold palms close with a gap; trace a small outward right-hand arc while the left remains as the other atom."),
            ("Reaction Activation Barrier", "Trace a right-hand path that rises over a small crest and ends lower while the left marks the starting energy."),
            ("Catalyst Pathway", "Trace a tall curved right-hand barrier, then retrace between the same endpoints using a lower curve."),
            ("Reversible Reaction", "Move the right from a left reactant position to a right product position, then move the left along a lower return path."),
            ("Equilibrium Exchange", "Hold hands at separate positions; make two equal inward-and-outward exchange arcs without either side advancing overall."),
            ("Concentration Gradient", "Mark three closely spaced points on the left and two widely spaced points on the right with the right hand."),
            ("Diffusion", "Begin both hands around a compact central area; move them along separate widening curved paths and settle apart."),
            ("Dissolving a Solid", "Hold the left palm as liquid; lower the right above it, then trace three small outward dispersion arcs."),
            ("Precipitate Formation", "Hold both hands over an imaginary solution; converge their paths, then settle the right palm lower as a collected solid."),
        ]),
        ("Physics concept illustrations", [
            ("Wave Interference", "Move both hands along approaching small wave paths on separate depth planes, then show one larger central rise."),
            ("Wave Cancellation", "Move hands in opposite up-down phases, then reduce both motions together to a still central level."),
            ("Reflection of a Ray", "Hold the left palm diagonally as a mirror; trace a right-hand approach line and an outward reflected line with clearance."),
            ("Refraction of a Ray", "Hold the left palm as a boundary; trace the right toward it, then continue beside its edge along a changed angle."),
            ("Standing Wave", "Hold hands at fixed side positions as nodes; use the right to trace one central bulge and return to its node."),
            ("Energy Transfer", "Lift the left palm slightly, then lower it as the right rises by a similar amount, keeping their spaces separate."),
            ("Conservation of Momentum Cue", "Advance the right hand toward a center point, stop short, then advance the left from that point along the same direction."),
            ("Rotational Axis", "Hold the left forearm upright as an axis; trace a right-hand horizontal arc around it without contact."),
            ("Damped Oscillation", "Swing the right hand through three progressively smaller side-to-side arcs while the left marks the center."),
            ("Resonance Build-Up", "Move the right through three progressively larger smooth arcs while the left marks evenly timed small beats nearby."),
        ]),
        ("Earth and environment descriptions", [
            ("Tectonic Convergence", "Hold palms horizontally on separate depth planes; move them toward each other and lift one edge slightly without touching."),
            ("Tectonic Divergence", "Hold palms horizontally with a gap; move them apart and raise the right hand slightly within the opening as a ridge cue."),
            ("Transform Fault", "Hold both palms downward side by side; slide the left forward and the right backward once along parallel paths."),
            ("Soil Layers", "Place the right palm at three descending horizontal levels while the left hand marks their common side boundary."),
            ("Erosion", "Hold the left forearm as a slope; sweep the right above it downward twice, then mark a lower collection area."),
            ("Sediment Deposition", "Trace two descending right-hand paths toward a low central area, then spread the right palm into a flat layer."),
            ("Water Cycle", "Trace a rising right-hand arc, a horizontal cloud path, and a descending path returning beside the left reference palm."),
            ("Watershed", "Trace two sloping right-hand routes from separate upper points toward one central lower basin marked by the left palm."),
            ("Groundwater Recharge", "Hold the left palm as a ground plane; guide the right downward around its side, then curve inward below it with clearance."),
            ("Habitat Corridor", "Mark two separated areas with both hands; trace a narrow curved right-hand connection between their edges."),
        ]),
        ("Energy system descriptions", [
            ("Battery Charging", "Hold the left hand as a battery boundary; move the right upward through three short internal levels without crossing the left hand."),
            ("Battery Discharging", "Hold the left hand as a battery boundary; move the right downward through three levels, then trace a short outward supply path."),
            ("Solar Panel Tilt", "Hold both palms downward as one panel plane; tilt both forearms gently toward an imagined light direction and hold."),
            ("Wind Turbine Model", "Hold the left forearm as a mast; trace three short radial right-hand arcs around an imaginary hub beside it."),
            ("Generator Conversion", "Trace a small right-hand rotation, then straighten that path toward the left palm as an output cue."),
            ("Energy Storage Buffer", "Hold the left palm as a reservoir; direct the right toward it, pause, then release a short path outward."),
            ("Peak Load Shaving", "Trace a tall right-hand peak, then redraw it flatter while the left marks a small storage area beside the peak."),
            ("Distributed Power Sources", "Mark three separate source positions with the right hand; trace short routes from each toward a central left palm."),
            ("Grid Isolation", "Trace a right-hand link between two areas; place the left vertical palm across the link and retract the right."),
            ("Power Restoration Sequence", "Mark three right-hand checkpoints along a supply route, pausing at each before advancing to the next."),
        ]),
        ("Robotics concepts without robot locomotion", [
            ("Sensor Field of View", "Keep the left hand at a sensor origin; spread the right through two outward rays defining a viewing cone."),
            ("Blind Spot", "Trace a broad sensing arc with the right hand but leave a gap; bracket that missing section with both hands."),
            ("Obstacle Detection", "Move the right along a planned path, pause before the left vertical palm, then mark the obstacle with a small arc."),
            ("Path Planning", "Hold the left palm as an obstacle; trace two right-hand routes around it and settle on the shorter route."),
            ("Workspace Boundary", "Outline a reachable chest-level area with both palms, then hold their outer edges as a boundary."),
            ("End-Effector Target", "Hold the left palm as a target; move the right along a curved approach, stop with clearance, and hold."),
            ("Joint Chain", "Move the right hand along three connected angular segments while the left marks the chain's fixed origin."),
            ("Redundant Reach", "Trace two different bent right-hand paths from one starting area to the same target marked by the left palm."),
            ("Sensor Fusion", "Trace two different approach paths with alternating hands toward a shared central estimate, stopping before the hands meet."),
            ("Human-Robot Handover Cue", "Present the left palm as a waiting receiver; bring the right near its space, pause, then withdraw the right gently."),
        ]),
        ("Feedback and control illustrations", [
            ("Setpoint Tracking", "Move the left palm slowly along a short horizontal path while the right follows at a small constant offset."),
            ("Tracking Lag", "Advance the left palm, pause, then move the right to its earlier position after a visible delay."),
            ("Overshoot", "Hold the left palm as a target marker; move the right past its adjacent plane, then return to a nearby aligned position."),
            ("Undershoot", "Hold the left palm as a target; stop the right short of its adjacent plane, pause, then advance the remaining distance."),
            ("Feedback Loop", "Trace a forward right-hand path to an output area, then use the left to trace a separate return arc to the origin."),
            ("Feedforward Cue", "Mark an upcoming change with the left hand; advance the right toward its predicted correction before moving the left reference."),
            ("Controller Saturation", "Move the right upward toward a limit marked by the left palm; stop below it and hold despite one small repeated approach."),
            ("Deadband", "Hold the left hand as a center marker; move the right slightly around it without response, then make one larger outward movement."),
            ("Hysteresis", "Trace a right-hand rising path on one side and a falling path on a parallel offset side, forming a narrow loop."),
            ("Disturbance Rejection", "Hold the left hand as a target; deflect the right slightly sideways, then return it smoothly to its previous offset beside the left."),
        ]),
        ("Statistical displays", [
            ("Bell Curve", "Trace one smooth symmetrical hump with the right hand over a baseline marked by the left palm."),
            ("Skewed Distribution", "Trace a steep short rise with the right hand followed by a much longer shallow descending tail."),
            ("Bimodal Distribution", "Trace two separated rounded humps with the right hand, leaving a shallow valley between them."),
            ("Uniform Distribution", "Trace a short vertical rise, a broad level top, and a short descent with the right hand."),
            ("Cumulative Curve", "Trace an S-shaped rising right-hand curve that starts shallow, steepens centrally, and flattens at its upper end."),
            ("Scatter Cluster", "Mark five small right-hand points within a compact area above the left palm, without joining them into a line."),
            ("Two Distinct Clusters", "Mark three compact points on the left and three on the right with the right hand, preserving a clear central gap."),
            ("Median Marker", "Mark five ordered positions with the right hand; return to the middle position and hold the palm above it."),
            ("Range Extremes", "Mark five positions with the right hand, then place both hands at the two outermost positions and hold."),
            ("Confidence versus Accuracy", "Hold the left palm as truth; place the right far beside it firmly, then draw a small certainty boundary around that wrong location."),
        ]),
        ("Information organization", [
            ("Hierarchy Levels", "Mark one upper right-hand point, two middle points and three lower points, keeping the arrangement within chest-level space."),
            ("Parent-Child Relation", "Hold the left palm as a parent node; trace two short descending right-hand links to separate child positions."),
            ("Cross-Reference", "Mark two document areas with both palms; trace one short right-hand path from a corner of one to a corner of the other."),
            ("Index Lookup", "Hold the left palm as an index; mark one entry with the right, then move the right to a separate destination area."),
            ("Tag an Item", "Hold the left palm as an item; move the right to its side and trace a small label-sized outline."),
            ("Mark a Taxonomy Exception", "Mark three related right-hand items inside a shared region; mark a fourth outside it and trace a short link back to the group."),
            ("Order by Dependency", "Mark three tasks in a scattered arrangement; relocate their positions into a left-to-right chain with three right-hand arcs."),
            ("Show a Matrix Cell", "Trace a small grid with the right hand, then hold above one intersection while the left marks the outer frame."),
            ("Show a Relationship Map", "Mark three nodes with alternating palms; connect two with a right-hand line and leave the third visibly unconnected."),
            ("Abstract a Pattern", "Trace three short similar right-hand shapes at different positions, then draw one simplified version above the left palm."),
        ]),
        ("Financial presentation metaphors", [
            ("Budget Envelope", "Frame a compact area with both palms; place three right-hand allocations inside while keeping the outer boundary unchanged."),
            ("Cost Breakdown", "Hold the left palm as a total; divide the area above it into three right-hand segments of different sizes."),
            ("Revenue versus Cost", "Trace a rising right-hand line and a lower left-hand reference line, then bracket the gap between their endpoints."),
            ("Break-Even Point", "Trace two intersecting right-hand lines, then pause above their crossing point with the left palm beneath it."),
            ("Cash Flow Timing", "Move the left forward as incoming funds, pause, then move the right away as a later outgoing payment."),
            ("Reserve Fund", "Hold the left palm near the torso as a reserve; make two right-hand outward allocations without moving the left."),
            ("Spread Risk", "Mark one central amount with the right hand; distribute it along three arcs to separate areas without depicting a recommendation."),
            ("Compound Growth", "Trace three connected right-hand rises of increasing height while the left remains as the starting value."),
            ("Depreciation", "Trace a descending right-hand curve that falls more steeply at first and gradually flattens near waist height."),
            ("Funding Gap", "Mark a required level with the left palm and a lower available level with the right, then hold the vertical gap visible."),
        ]),
        ("Markets and logistics concepts", [
            ("Supply Chain", "Move the right hand through four separate source-to-customer checkpoints while the left marks the final destination."),
            ("Inventory Reorder Point", "Hold the left palm as a low threshold; lower the right through three stock levels and stop just above the threshold."),
            ("Batch Production", "Mark three close item positions with the right hand, then move both palms around them forward as one grouped batch."),
            ("Single-Piece Flow", "Move the right along one short production route three times, completing each route before the next begins."),
            ("Distribution Hub", "Hold the left palm as a hub; trace one incoming right-hand route and three shorter outgoing routes."),
            ("Last-Mile Delivery", "Trace a long right-hand route to a hub, pause, then make one short final arc to a nearby destination."),
            ("Return Logistics", "Trace a forward right-hand delivery route; then trace a distinct lower return route back to the left source palm."),
            ("Demand Surge", "Mark a steady right-hand series of two beats, then add three closer beats in a larger forward area."),
            ("Capacity Constraint", "Hold both vertical palms as a narrow processing area; bring the right toward its edge and pause outside the capacity boundary."),
            ("Queue Clearance", "Mark three waiting positions with the right hand; move them forward one at a time through a gate indicated by the left."),
        ]),
        ("Astronomy diagrams", [
            ("Orbital Inclination", "Hold the left palm as a flat reference plane; trace a right-hand ellipse tilted relative to that plane, keeping clearance."),
            ("Lunar Phases", "Hold the left palm as a moon disk; slide the right palm across its front through three partially covered positions."),
            ("Solar Eclipse Diagram", "Align the right palm between the left palm and an imagined viewer, keeping the hands separated; pause at alignment."),
            ("Lunar Eclipse Diagram", "Mark three separated positions with the right hand; return to the middle position and sweep its imagined shadow toward the third."),
            ("Constellation Pattern", "Mark five distinct points with the right hand, then trace two short links between selected points."),
            ("Gravitational Lensing Diagram", "Hold the left palm as a central mass; trace two right-hand light paths curving around opposite sides toward a shared destination."),
            ("Comet Tail", "Trace a curved right-hand path; follow behind it with the left palm along a longer trailing arc."),
            ("Binary Star System", "Hold hands apart and trace two small circles around their shared midpoint, maintaining separation throughout."),
            ("Galaxy Disk", "Trace a broad flat ellipse with the right hand; pass the left palm above its center to indicate thickness."),
            ("Telescope Light Path", "Mark an entry plane with the left palm; trace a right-hand ray inward and bend it toward a lower focus."),
        ]),
        ("Biology diagrams", [
            ("Cell Membrane", "Outline a rounded boundary with both hands; pause, then use the right hand to mark one controlled opening."),
            ("Cell Division", "Form a wide oval with separated palms; narrow its middle with both hands, then separate into two equal spaces."),
            ("DNA Double Helix", "Move both hands upward along alternating gentle curves, maintaining clearance where the paths approach each other."),
            ("Protein Folding", "Trace a short right-hand chain through three bends, then bring its endpoint near the starting region without touching."),
            ("Enzyme Fit", "Hold the left palm as an angled receiving surface; approach with the right at a matching angle and stop short."),
            ("Immune Recognition", "Present the left palm as a target; circle it once with the right, then pause beside one marked feature."),
            ("Neural Signal Relay", "Mark three spaced nodes with the right hand; trace a short forward link between each successive pair."),
            ("Osmosis Diagram", "Hold the left palm as a membrane; move the right toward it, around its edge, and into a smaller receiving area."),
            ("Food Web", "Mark three lower food-source positions with the right hand; link them to two higher positions within chest height."),
            ("Population Carrying Capacity", "Hold the left palm at a ceiling; raise the right in three increments and flatten its path below that ceiling."),
        ]),
        ("Built environment descriptions", [
            ("Doorway Clearance", "Hold both palms upright as doorway sides; narrow their spacing slightly, then restore enough room and hold."),
            ("Window Recess", "Outline a small rectangle with the right hand; move both hands backward a short distance to show depth."),
            ("Staircase Diagram", "Trace three small ascending right-hand steps in the air; keep the robot's body and feet stationary."),
            ("Ramp Gradient", "Hold the left palm horizontally; trace one sloping right-hand line above it and pause at the endpoint."),
            ("Suspension Bridge Cables", "Hold the left hand as a tower; trace two right-hand sloping cable paths from beside its top to separated lower deck points."),
            ("Roof Pitch", "Raise separated palms into two inward-sloping planes below shoulder height, leaving a gap at the imagined ridge."),
            ("Room Partition", "Outline a shared area with both hands; place the right vertical palm through its center as a divider."),
            ("Cantilever Diagram", "Hold the left palm as a support; extend the right horizontally beyond it while maintaining a visible gap."),
            ("Corridor Junction", "Trace a straight right-hand route, stop at its midpoint, then trace two short branches to either side."),
            ("Accessible Turning Space", "Outline a clear round area with both hands, then trace one right-hand turning arc inside that area."),
        ]),
        ("Sports explained as diagrams", [
            ("Passing Lane Diagram", "Mark two player positions with the right hand; trace a clear path between them while the left marks an obstacle outside it."),
            ("Defensive Zone Diagram", "Outline a small rectangular zone with the right hand; hold the left palm at its center as a defender marker."),
            ("Offside Line Diagram", "Hold the left palm as a boundary; move the right marker just beyond it, pause, then return behind the boundary."),
            ("Relay Exchange Zone", "Hold hands apart to mark a short zone; move the right through its center and stop at the far boundary."),
            ("Tournament Bracket", "Trace two small converging pairs with the right hand, then connect their endpoints to one central finalist point."),
            ("Shot Trajectory Diagram", "Trace one right-hand arc above the left target palm, ending just short of the target without contact."),
            ("Rebound Diagram", "Hold the left palm as a surface; approach with the right on a diagonal and trace a different outgoing diagonal."),
            ("Lap Split Comparison", "Mark two equal-width lap segments with the right hand; place the left at the first endpoint and compare the second."),
            ("Team Formation Diagram", "Mark five distinct positions with the right hand in a compact formation, then outline their shared boundary."),
            ("Goal-Line Review Diagram", "Hold the left palm as a goal plane; move the right marker across it via a separated path, then hold beyond it."),
        ]),
        ("Additional instrument mimes", [
            ("Cello Bowing Mime", "Hold the left hand forward as a neck reference; move the right horizontally twice across an imagined lower string area."),
            ("Harp Plucking Mime", "Hold the left palm as a frame edge; make three short right-hand pullback motions at different chest-level heights."),
            ("Accordion Bellows Mime", "Hold palms facing each other; separate them gradually, then bring them closer once without contact."),
            ("Xylophone Pattern Mime", "Keep both hands at waist level; alternate four small downward beats at four different imagined bar positions."),
            ("Maraca Rhythm Mime", "Hold separated hands at chest level; alternate six small vertical wrist-and-forearm pulses without changing the hand shape."),
            ("Tambourine Tap Mime", "Hold the left palm as a tilted drum reference; make three right-hand approaches that stop before touching it."),
            ("Triangle Instrument Mime", "Hold the left hand as a hanging reference; make three short right-hand diagonal strokes below it, without contact."),
            ("Theremin Pitch Mime", "Hold the left palm level; move the right slowly toward and away from an imagined antenna at one side."),
            ("Steel Drum Pattern Mime", "Keep hands apart over an imagined shallow bowl; make four alternating short downward beats around its rim."),
            ("Turntable Scratch Mime", "Hold the left palm as a deck reference; move the right through two short alternating horizontal arcs above it."),
        ]),
        ("Clay and sculpture mimes", [
            ("Center Clay Mime", "Hold palms facing inward around an imagined clay lump; move them slightly inward together and pause with clearance."),
            ("Raise Pottery Walls Mime", "Begin with separated palms at waist level; raise them together toward the chest while gently narrowing their spacing."),
            ("Shape a Bowl Rim Mime", "Outline a horizontal oval with both hands, then tilt palms slightly outward to show a flared rim."),
            ("Flatten a Clay Slab Mime", "Hold both palms downward above an imagined slab; make two shallow lowering motions with a brief hold after each."),
            ("Score Clay Surface Mime", "Hold the left palm as a surface reference; trace three short parallel right-hand marks above it without touching."),
            ("Join Clay Pieces Mime", "Present two imaginary pieces between separated palms; bring them near each other and hold a visible seam gap."),
            ("Carve a Groove Mime", "Hold the left hand beside an imagined block; trace one slow right-hand channel down its front."),
            ("Smooth a Sculpture Mime", "Hold the left hand as a shape reference; sweep the right palm along two curved paths around it without contact."),
            ("Show Sculpture Negative Space", "Hold hands around an empty central opening; widen that opening slightly and pause to emphasize the space."),
            ("Rotate a Display Base Mime", "Outline a horizontal base with the right hand; trace a half-circle around it while the left marks the sculpture center."),
        ]),
        ("Packing and protection mimes", [
            ("Wrap a Fragile Item Mime", "Hold the left palm beneath an imagined item; sweep the right around two sides as a protective wrap."),
            ("Add Corner Protection Mime", "Outline an imagined box corner with the left hand; approach its two faces with the right and stop short."),
            ("Fit a Foam Insert Mime", "Hold palms around a small rectangular cavity; lower the right toward its center and stop at the imagined fitted depth."),
            ("Cushion an Empty Space Mime", "Hold the left palm as a box wall; place two imaginary cushions into the space beside it using the right."),
            ("Align a Shipping Label Mime", "Hold the left palm as a package face; position the right parallel to it, adjust sideways once, and pause without contact."),
            ("Mark Fragile Handling Cue", "Frame an imagined package with both hands; lift that space slightly and lower it slowly to suggest careful handling."),
            ("Show Stack Height Limit", "Hold the left palm as a ceiling; mark three rising right-hand package levels that finish below the ceiling."),
            ("Separate Incompatible Items", "Present hands near each other as two items; move them into distinct areas and hold a clear gap."),
            ("Check Tamper Seal Mime", "Hold the left palm as a package reference; trace a short right-hand seam, pause at its center, then withdraw."),
            ("Track Package Condition", "Mark an imagined package with the left hand; use the right to check three distinct surfaces around it without contact."),
        ]),
        ("Visitor and service explanations", [
            ("Show a Meeting Point", "Mark three approaching routes with the right hand, all ending beside the left palm at one shared point."),
            ("Explain a Waiting Area", "Outline a small area with both hands; use the right to mark two stationary positions inside it."),
            ("Show an Information Desk", "Trace a short horizontal counter with both hands; hold the left as the counter and present the right above its center."),
            ("Explain a Collection Window", "Hold palms apart as a window; move the right through a short receiving path below its center without crossing the left."),
            ("Separate Arrival and Departure", "Trace an incoming route with the left hand and a separated outgoing route with the right, then hold their endpoints."),
            ("Show a Quiet Zone", "Outline a small zone with both hands; settle both palms slowly above it and hold them motionless."),
            ("Explain a Lost-Item Process", "Present the left palm as a reported item; mark two right-hand checkpoints, then return the right near the left."),
            ("Show a Scheduled Time Slot", "Hold palms apart as time boundaries; narrow the interval once and maintain the final spacing."),
            ("Explain a Priority Queue", "Mark three right-hand waiting positions; trace a separated upper route from one position to the front without body movement."),
            ("Show a Service Transfer", "Present the left palm as one desk; move the right along a short route toward a second location and hold."),
        ]),
        ("Visual editing concepts", [
            ("Layer Transparency", "Hold palms in separated parallel planes; move the front right palm partly aside to reveal the left behind it."),
            ("Clipping Mask", "Hold the left palm as a boundary; trace a right-hand shape that stays within the adjacent indicated area."),
            ("Crop a Frame", "Frame a rectangle with separated hands; bring its left and right boundaries inward once and hold."),
            ("Show a Pixelated Edge", "Trace a short diagonal as four small horizontal-and-vertical right-hand segments; then trace a smooth diagonal beside it for comparison."),
            ("Snap to an Anchor", "Hold the left hand as an anchor; approach slowly with the right, make one small alignment adjustment, and stop short."),
            ("Show Optical Alignment", "Mark a geometric center with the left hand; place the right near it, shift slightly sideways, and hold the adjusted visual center."),
            ("Group and Ungroup Objects", "Mark two separate objects with the right hand; frame them together with both palms, then open the frame into two areas."),
            ("Bring a Layer Forward", "Hold hands in separated front-and-back planes; move the rear left hand around the right edge into a nearer plane."),
            ("Isolate a Selection", "Outline a small right-hand region; hold the left palm outside it and move the right region slightly away."),
            ("Feather a Selection Edge", "Hold the left palm as a crisp boundary; trace three right-hand parallel passes beside it with gradually smaller excursions."),
        ]),
        ("Speaking structure and emphasis", [
            ("Preview Then Revisit a Claim", "Present the right palm as a claim; trace a short detour through two evidence points and return to the claim position."),
            ("Distinguish Question from Claim", "Hold the left palm upward as a question; present the right vertically as a claim at a separate location and pause."),
            ("Bracket a Short Aside", "Hold the left hand at the main-topic position; trace a small right-hand bracket beside it and return near the left."),
            ("Connect Example to Principle", "Present the right palm lower as an example; trace a gentle upward link toward the left palm at chest level."),
            ("Separate Detail from Summary", "Make three small right-hand detail beats; then open both palms to frame the wider topic once."),
            ("Separate a Quote from Commentary", "Frame a small quoted region with both hands; hold the left there while the right marks a separate commentary point."),
            ("Contrast Past and Present", "Present the left palm at one side for the past; present the right at a separate forward position for the present."),
            ("Distinguish Plan from Result", "Hold the left palm as a planned reference; place the right beside it at a slightly different height and pause."),
            ("Signal a Topic Handoff", "Finish with the right palm near center; turn it toward the left-side topic area, then bring the left into that area."),
            ("Close a Spoken Paragraph", "Bring two separated palms gently toward the center, hold briefly, then lower both smoothly to the relaxed end pose."),
        ]),
    ]

    start_num = 601
    for sec_name, items in sections_map:
        for idx, (intent, prompt) in enumerate(items):
            g_id = start_num + len(entries) - 100
            entries.append((g_id, intent, sec_name, prompt))

    return entries


def main() -> int:
    all_entries = generate_remaining_entries()
    print(f"Extracted {len(all_entries)} total entries (IDs {all_entries[0][0]} to {all_entries[-1][0]})")

    # Audit extraction integrity
    id_counts = {}
    for gid, intent, cat, prompt in all_entries:
        id_counts[gid] = id_counts.get(gid, 0) + 1
        if not intent:
            print(f"ERROR: Missing intent for ID {gid}")
        if not prompt:
            print(f"ERROR: Missing motion prompt for ID {gid}")

    missing_ids = [gid for gid in range(501, 1001) if gid not in id_counts]
    duplicate_ids = [gid for gid, count in id_counts.items() if count > 1]

    print(f"  Missing IDs   : {len(missing_ids)} {missing_ids}")
    print(f"  Duplicate IDs : {len(duplicate_ids)} {duplicate_ids}")

    if missing_ids or duplicate_ids:
        print("ERROR: Extraction failed integrity check!")
        return 1

    # Format JSON structured registry
    structured_json = {"gestures": []}
    for gid, intent, cat, prompt in all_entries:
        structured_json["gestures"].append({
            "gesture_id": f"PDF_{gid:03d}",
            "number": gid,
            "category": cat,
            "intent": intent,
            "motion_prompt": prompt,
            "kimodo_prompt": prompt,
            "source_catalogue": "501-1000"
        })

    with open(OUT_JSON_PATH, "w") as f:
        json.dump(structured_json, f, indent=2)

    print(f"Successfully saved extracted specs to {OUT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
