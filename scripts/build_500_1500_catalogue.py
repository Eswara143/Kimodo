#!/usr/bin/env python3
"""Build complete gesture catalogue JSON for IDs 500 to 1500 (1001 total gestures).

Combines:
- ID 500 ("Heavy Object" from PDF_500)
- IDs 501–1000 from gesture_library_501_1000.json (with spoken_response added)
- IDs 1001–1500 conversational dialogue & spoken response catalogue entries

Outputs gesture_library_500_1500.json.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_501_1000 = os.path.join(ROOT, "gesture_library_501_1000.json")
JSON_250 = os.path.join(ROOT, "gesture_library_250.json")
OUT_JSON = os.path.join(ROOT, "gesture_library_500_1500.json")

# Data catalogue for 1001..1500 categories and entries
RAW_CATALOGUE_1001_1500 = """
### Conversational Repair & Clarification
1001 | Could you please say that again in another way? | Present the right palm upward at chest level, draw it slightly back, then make one short outward arc and hold.
1002 | Could you please tell me again? I am unable to hear you clearly. | Hold the left palm steady near the chest; raise the right palm beside it, tilt outward once, then pause.
1003 | There is just one small correction I would like to make. | Frame a small area with both palms; move the right palm sideways to a corrected position while the left remains fixed.
1004 | Let me come back to the point we were discussing. | Hold the left palm as the active topic; move the right from a side detour back beside it and hold.
1005 | Could you please explain it a little more slowly? | Hold both palms forward and move them downward together in two slow, even settling motions.
1006 | Could you please give me a shorter answer? | Frame a wide horizontal space with both hands, then bring the side boundaries inward and hold the smaller frame.
1007 | Could you please explain that in a little more detail? | Hold both palms around a small central space, then widen the frame gradually while keeping the chest and body still.
1008 | This is the part that I did not understand clearly. | Present the left palm as the topic; circle one small area above it with the right hand and hold there.
1009 | Yes, that correction is clear now. | Move the right palm from the corrected side position to center, then make one small downward confirmation beat.
1010 | Okay, I understand now. Please continue. | Hold both palms briefly still near the chest, then send the right palm forward in one smooth continuation arc.

### Turn-Taking & Listening Cues
1011 | I have finished my point. You can continue. | Finish with the right palm centered, rotate it upward toward the listener, then retract it to a relaxed chest-level pose.
1012 | I would like to add something here. | Raise the right palm from waist to chest height and hold it upright without moving the head or torso.
1013 | Please give me a moment to finish this point. | Keep the left palm open toward the listener while the right makes one small forward beat, then hold both steady.
1014 | Please continue. I am listening. | Present both palms upward, then move the right palm gently forward once and hold.
1015 | Yes, I am following what you are saying. | Keep the left palm relaxed near the chest; make two small downward right-hand beats synchronized and controlled.
1016 | I understand. Please continue. | Make one compact upward-and-downward pulse with the right forearm while the left remains still.
1017 | That is all I wanted to say for now. | Bring both palms toward the center from separated positions, pause, then lower only the forearms slightly.
1018 | Give me a moment to think before I answer. | Raise both palms to equal chest height, hold completely still for a brief beat, then relax a little.
1019 | We can discuss this point together. | Present both palms side by side, move them forward together a short distance, and hold equal spacing.
1020 | Let us hear the other person on this point. | Present the right palm toward one listener position, sweep it horizontally to a second position, then hold there.

### Rhetorical Emphasis & Delivery
1021 | This is the key phrase I want to emphasize. | Hold the left palm as a phrase marker; sweep the right palm horizontally beneath it in one clean line.
1022 | This one word is especially important. | Keep the left palm relaxed; make one sharp but controlled right-hand downward beat at chest height.
1023 | There are two important points to remember. | Mark a left point with the left palm and a right point with the right palm using two separate downward beats.
1024 | Now I am coming to the most important point. | Move the right palm through three progressively higher chest-level positions, pausing briefly at the final one.
1025 | I want to make that claim a little more carefully. | Present the right palm firmly forward, then reduce its reach halfway and turn the palm slightly upward.
1026 | Let me add one small side note here. | Keep the left palm at the main-topic position; move the right to a small side space and make one gentle beat.
1027 | There is an important contrast between these two points. | Hold palms at separate left and right positions; advance the left once, then the right once, preserving the gap.
1028 | And that is the key point. | Hold both palms still, then make one quick outward opening motion and stop with arms comfortably bent.
1029 | Let me explain this more slowly. | Move the right palm through two forward beats with visibly longer pauses and smooth deceleration between them.
1030 | That is the main point I want to leave with you. | Bring both palms inward toward center, make one final small downward beat together, then return to the relaxed pose.

### Negotiation & Consensus
1031 | Can we choose a middle option that works for both sides? | Present left and right palms at separate positions, then move both halfway toward center and hold without touching.
1032 | I cannot accept this particular condition. | Hold the left palm as the proposal; place the right palm vertically beside one edge and move that edge outward.
1033 | I can accept this condition. | Hold the left palm as the proposal; bring the right palm beneath one section and make a small upward support motion.
1034 | We can treat these two points as one agreement. | Mark two side points with alternating hands, then frame both together with a wider two-palm boundary.
1035 | These points are agreed, but these points are still open. | Place the left palm forward as agreed; keep the right palm upward at a separate side position for unresolved items.
1036 | Let us meet somewhere in the middle. | Mark two outer positions with both hands, then move both palms to equal distances from a central gap.
1037 | I am withdrawing this condition. | Present the right palm at a side condition point, then draw it smoothly back toward the chest-level neutral zone.
1038 | Let us come back to the goal we both share. | Mark two separated starting points with both hands and trace short converging arcs toward one shared forward point.
1039 | It looks like we are close to an agreement. | Bring both upward palms closer to center, stop with a visible gap, and make one small synchronized settling pulse.
1040 | Yes, we have reached a final agreement. | Align both palms side by side at chest height, advance them together once, then hold still.

### Decision Trade-Offs
1041 | Let us compare the benefit with the cost. | Hold the left palm slightly higher and the right slightly lower; alternate one small lift from each, then level them.
1042 | I would choose the option with lower risk. | Mark two options with both palms; trace a smaller right-hand uncertainty arc over one and move that option forward.
1043 | I would choose the faster option. | Mark two routes with the right hand; retrace the shorter route once while the left marks its endpoint.
1044 | I would choose the simpler option. | Outline two imagined structures; keep one broad, then redraw the other as a shorter two-segment path.
1045 | This shows the trade-off between the two choices. | Hold the left as one axis reference; trace a smooth descending right-hand curve across the chest-level workspace.
1046 | This trade-off is not acceptable. | Present two palms at unequal levels, then place the left vertically between them as a firm boundary.
1047 | I would prefer the more flexible option. | Hold two side options; keep one fixed with the left while the right traces two alternate short paths from its option.
1048 | I would prefer the more stable option. | Hold one palm as the chosen position and make two small disturbances with the other that return to the same offset.
1049 | Let us postpone this decision for now. | Present two upward palms equally, begin a forward selection motion, stop halfway, and hold both unresolved.
1050 | I am ready to commit to this choice. | Move the chosen right palm from a side option to center and make one short downward confirmation beat.

### Teaching & Scaffolding
1051 | This is the learning goal for this step. | Frame a small chest-level target area with both hands, then place the right palm forward at its center.
1052 | Let me demonstrate the first step. | Hold the left palm as a starting point; move the right along one clear straight segment and pause at the endpoint.
1053 | Let me demonstrate the two-step sequence. | Move the right through two connected chest-level segments with a brief hold at the intermediate point.
1054 | Let us break this task into smaller parts. | Frame one large area with both hands, then use the right palm to mark three smaller regions inside it.
1055 | Now let us connect these parts back to the whole task. | Mark three separate regions with the right hand, then widen both palms to frame them as one larger unit.
1056 | Here is a small hint to help you continue. | Hold the left palm as the problem; trace only the first half of a right-hand solution path and stop.
1057 | Now I will show you the next step. | Keep the left as the current step; move the right from behind it to the next forward position and hold.
1058 | Is everything clear so far? | Present both palms upward near the chest and hold them open without repeated motion.
1059 | Here is an example of the pattern. | Trace one short shape with the right hand, then repeat the same path at a second nearby position.
1060 | Let me summarize what we learned. | Make three compact right-hand beats for key points, then open both palms around one central summary area.

### Attention & Referencing
1061 | Please focus on the center. | Open both palms from the sides toward a shared central area and hold them framing that space.
1062 | Please look at the panel on the left. | Hold the right palm near center; sweep it leftward and stop beside a left-side reference marked by the left hand.
1063 | Please look at the panel on the right. | Hold the left palm near center; sweep the right palm rightward and stop at the right-side reference.
1064 | Please focus on this small region. | Frame a compact rectangular space with both palms and hold its boundaries steady.
1065 | Please focus on this larger region. | Begin with palms near center, widen them symmetrically to frame a broad area, then hold.
1066 | I am referring to the point I mentioned earlier. | Hold the left palm as the current point; curve the right hand backward within the workspace to an earlier side position.
1067 | I am referring to the point we will discuss later. | Hold the left palm as now; trace the right forward to a farther chest-level position and pause.
1068 | I am referring to these two items together. | Mark one item with each palm, then move both slightly inward while preserving separation.
1069 | I am referring to this item only. | Mark two positions with both hands, retract the left reference, and keep the right fixed on the selected item.
1070 | Now please bring your attention back to the center. | Bring the active right palm from a side reference back to the centered neutral talking position.

### Memory & Recall
1071 | I remember a recent detail that is relevant here. | Hold the left palm as the present; curve the right hand to a nearby rearward workspace point and make one small beat.
1072 | I remember an older detail that may help here. | Hold the left as the present; trace a longer curved right-hand path farther to the side-rear workspace and pause.
1073 | Let me recall the sequence in the correct order. | Mark three ordered positions from left to right with the right hand, then retrace them in the same order.
1074 | I forgot one step in that sequence. | Mark three ordered positions, return to the middle one, circle its empty area once, and leave that gap visible.
1075 | I remember the missing step now. | Hold a visible gap between two marked positions, then place the right palm into the gap and hold.
1076 | These two memories are connected. | Mark two separated points with alternating hands, then trace one curved right-hand connection between them.
1077 | These two memories are similar, but they are separate. | Present two nearby palms, move them farther apart horizontally, and hold them at equal height.
1078 | Let me correct what I remembered incorrectly. | Hold the left palm as the remembered version; move the right from beside it to a corrected lower position.
1079 | Let us remember this key detail. | Frame a small space with both palms near the chest, narrow the frame slightly, and hold it steady.
1080 | Let me bring back that key detail. | Hold both palms around a small stored area, then move the right palm outward from its center to the listener.

### Goals & Progress Tracking
1081 | Let us set a new goal. | Frame a target-sized area with both hands; move the right palm to its center and hold firmly.
1082 | We have made some early progress. | Hold the left palm as start; move the right one-quarter of a short forward path and pause.
1083 | We are about halfway to the goal. | Hold the left as start; move the right to the midpoint of a marked path and hold.
1084 | We are close to completing the goal. | Hold the left as start; move the right close to the final marker but stop short with a visible gap.
1085 | We have completed the goal. | Move the right to the final marker beside the left reference, then make one small downward confirmation beat.
1086 | Our progress has stopped for now. | Move the right along a short path, stop midway, then make two tiny forward attempts without advancing.
1087 | We have moved backward from our earlier progress. | Move the right forward along a path, pause, then retrace part of the same path backward.
1088 | Let us reset the goal and start again. | Hold the right at the current goal position; sweep it back to center and frame a new target with both hands.
1089 | Let us raise the target. | Hold the left palm as the old target; place the right at a higher chest-level position and hold.
1090 | Let us lower the target. | Hold the left as the old target; place the right at a lower chest-level position and hold.

### Constraints & Requirements
1091 | This is a hard constraint that we must not break. | Hold the left palm vertically as a boundary; move the right toward it and stop clearly before crossing.
1092 | This is a soft constraint that we should try to follow. | Hold the left as a boundary; approach with the right, slow gradually, and stop with a larger flexible gap.
1093 | This is the minimum requirement. | Hold the left palm as a lower threshold; bring the right upward to just above it and hold.
1094 | This is the maximum allowed requirement. | Hold the left palm as an upper threshold; raise the right toward it and stop just below.
1095 | This is the allowed range. | Place palms at lower and upper chest-level boundaries and keep the open interval visible between them.
1096 | This range is not allowed. | Frame a central interval with both hands, then move both palms outward to emphasize exclusion from that space.
1097 | We need to add one more requirement. | Hold the left palm as the current set; bring the right from the side into a new adjacent requirement position.
1098 | We can remove this requirement. | Hold both palms as two requirements; retract the right one smoothly while the left remains as the retained item.
1099 | Let me show how this requirement leads to the outcome. | Hold the left as a requirement; trace a right-hand path from it to one forward outcome position.
1100 | These two requirements are in conflict. | Present two vertical palms facing inward on separate paths and stop before they overlap.

### Workflow Exceptions & Recovery
1101 | Here is the normal flow. | Trace three connected forward checkpoints with the right hand while the left marks the starting point.
1102 | Here is an exception branch. | Trace a short main route, then branch the right hand diagonally to a separate side endpoint.
1103 | Let me show you return from an exception. | Hold the left palm on the main route; curve the right hand from the side branch back beside it.
1104 | Let me show you skip an optional step. | Mark three route points, arc the right hand from the first over the middle to the third.
1105 | Let me show you retry a failed step. | Trace a short right-hand step, retract to its start, then repeat the same step once more.
1106 | I am escalating an exception. | Hold the left palm as the blocked step; raise the right to a higher decision position and hold.
1107 | Let me show you fallback to a safe path. | Mark two routes with the right hand; abandon the upper route and trace the lower route to completion.
1108 | Let me show you pause the workflow. | Move the right through one route segment, place the left palm vertically at the next checkpoint, and hold.
1109 | Let me show you resume the workflow. | Keep the left as the paused checkpoint; lower it slightly and move the right through the next segment.
1110 | Let us close the recovery loop. | Trace a side detour with the right hand, curve back to the main route, and finish at the original endpoint.

### Version Control Concepts
1111 | Let us create a new branch. | Trace one right-hand main line, then split into a second diagonal route while the left marks the branch point.
1112 | Let us switch branches. | Mark two parallel route positions; move the right from one to the other with a short sideways arc.
1113 | Let us merge two branches. | Trace two separated inward paths with alternating hands and finish them beside one common central endpoint.
1114 | Let us resolve a merge conflict. | Hold palms facing each other as two conflicting changes, rotate them to matching orientation, then align side by side.
1115 | Let us save this change as a commit. | Present a small change above the left palm, then make one compact downward right-hand beat to lock the state.
1116 | Let us revert this change. | Move the right from a prior position forward, pause, then retrace exactly back to the prior position.
1117 | Let us apply this selected change to the current branch. | Mark three changes with the right hand, select the middle position, and move only that marker to a new branch area.
1118 | Let us mark this release with a version tag. | Trace a short release line with the right hand, then outline a small label-sized box beside its endpoint.
1119 | Let us compare two revisions. | Hold palms at two separated revision positions; alternate one forward presentation beat from each.
1120 | Here is the commit history. | Mark four ordered checkpoints with the right hand along a shallow horizontal timeline and pause at the latest.

### Debugging Concepts
1121 | Let me show you how locate a bug works. | Trace a short process path with the right hand, stop at one point, and circle that location once.
1122 | Let me show you how reproduce a bug works. | Trace the same three-step right-hand path twice, ending both times at the same marked fault point.
1123 | Let me show you how narrow the fault location works. | Frame a broad process area with both hands, then narrow the frame around one small right-side region.
1124 | Let me show you how test a fix works. | Hold the left palm as the bug location; route the right around it through a new path and pause at the outcome.
1125 | Let us compare before and after fix. | Present the left palm as before and the right as after at separate positions, then alternate one small beat each.
1126 | Here is an intermittent failure. | Move the right through a normal path twice; on the third pass, stop early at the marked fault point.
1127 | Let me show you how trace an error backward works. | Hold the right palm as the observed error; move the left backward through two causal checkpoints toward the source.
1128 | Let us add a diagnostic check. | Trace a main process line, then place the left palm vertically at its midpoint as an inspection gate.
1129 | Let us remove a false lead. | Mark two suspect areas; retract one palm to neutral while keeping the confirmed suspect position fixed.
1130 | Let me show you how confirm the root cause works. | Trace two backward paths to one common origin and hold both palms framing that source point.

### API Integration Concepts
1131 | Let us send an API request. | Hold the left palm as a client; move the right along a straight path toward a separated server position.
1132 | We received an API response. | Hold the right as the server; move the left along a return path toward the client position.
1133 | Here is the request and response pair. | Trace one outward route with the right hand and a lower return route with the left.
1134 | Let me mark an API endpoint. | Trace a short route with the right hand and finish at a clearly held endpoint beside the left palm.
1135 | Here is an authentication header. | Hold the left palm as a request; position the right above it as a small attached header band.
1136 | Here is a request payload. | Frame a compact message area with both palms, then move that framed space forward together.
1137 | Here is a timeout. | Start a request path with the right hand, stop before the endpoint, hold, then retract slowly.
1138 | Here is a retry. | Start and abort one request path, return to origin, then complete the same path once.
1139 | Here is the rate limiting. | Make three quick right-hand request beats toward the left boundary; block the fourth with the left vertical palm.
1140 | Here is an API gateway. | Trace two incoming routes with alternating hands toward one central gate marked by the left palm.

### Database Concepts
1141 | Let us insert a record. | Frame a table area with the left hand; move the right into one empty row position and hold.
1142 | Let us update a record. | Mark one row position with the left; shift the right marker within that row to a new value position.
1143 | Let us delete a record. | Mark a row with the right palm, then retract it downward out of the framed table area.
1144 | Let us query a table. | Hold the left palm as a table; sweep the right across it and stop over one matching region.
1145 | Let us filter records. | Mark four row positions with the right hand, then keep only two framed between the palms.
1146 | Let us sort records. | Mark three uneven positions, then relocate them with right-hand arcs into a rising left-to-right order.
1147 | Let us join two tables. | Frame a left table and right table with separate palms, then trace one central overlapping relation zone.
1148 | Here is a primary key. | Mark three row positions; hold the right palm uniquely above one row while the left frames the table.
1149 | Here is an index. | Hold the left as the table; mark a compact ordered side list with three short right-hand beats.
1150 | Here is a transaction. | Move the right through three operation points, then finish with one firm confirmation beat at the end.

### Data Structure Concepts
1151 | Let us push one item onto the stack. | Hold the left palm as a stack base; place the right at a new higher layer and pause.
1152 | Let us pop the top item from the stack. | Hold the right at the top layer of an imagined stack, lift it away, and return to neutral.
1153 | Let us add one item to the back of the queue. | Frame a horizontal queue with both hands; move the right into the rear position and hold.
1154 | Let us remove the item at the front of the queue. | Hold the left at the front position; move that marker outward while the right keeps the remaining queue visible.
1155 | Here is the linked-list chain. | Mark four right-hand node positions in a row and connect adjacent positions with short curved links.
1156 | Here is the binary-tree split. | Hold the left palm as a parent node; trace two descending right-hand branches to separated child positions.
1157 | Here is how the value maps into a hash bucket. | Mark one key position with the right, trace a short arc to one of three bucket positions, and hold.
1158 | This is the top element of the heap. | Frame a compact triangular arrangement with both hands; hold the right palm above its highest central element.
1159 | Here is the circular buffer. | Trace a chest-level oval with the right hand and mark three evenly spaced positions around it.
1160 | Let us reorder the priority queue. | Mark three queue items, then move one side item forward to the front with an arcing right-hand path.

### Algorithm Concepts
1161 | Let me demonstrate a linear search. | Mark five positions left to right; move the right palm over each sequentially until stopping at the target.
1162 | Let me demonstrate a binary search. | Frame a sorted interval with both hands; mark the midpoint, discard one half, then mark the new midpoint.
1163 | Let me demonstrate one bubble-sort pass. | Mark four ordered positions; swap one adjacent pair with crossing but separated hand arcs.
1164 | Let me demonstrate one insertion-sort step. | Hold three ordered positions; move one new right-hand marker into the correct gap between two positions.
1165 | Let me demonstrate one selection-sort step. | Mark four items; select the lowest position, then move it to the leftmost slot.
1166 | Let us divide the problem and solve the smaller parts. | Frame one large region, split it into two side regions, then present both regions separately.
1167 | Here is the dynamic-programming table. | Trace a small grid with the right hand, then fill three cells in an ordered path.
1168 | Let us take the best immediate choice at this step. | Mark three available options; move the nearest right-hand option immediately to the chosen center position.
1169 | That path did not work, so let us go back and try another one. | Trace a two-segment route forward, pause at a dead end, then retrace one segment and branch elsewhere.
1170 | Let us save this result so we can reuse it later. | Mark a solved result beside the left palm, later return the right directly to that stored position without retracing the process.

### Distributed Systems Concepts
1171 | Let us replicate data. | Hold the left palm as source data; trace three right-hand copy paths to separate node positions.
1172 | Here is the leader and followers. | Hold the right palm slightly forward as leader; place three follower positions behind with small left-hand beats.
1173 | Let us choose a leader for the group. | Mark three peer positions; raise one selected palm to a distinct higher chest-level position.
1174 | Let us run one consensus round. | Mark three node positions, move each toward a common central value, and hold aligned around the center.
1175 | The network is split into separate partitions. | Frame one cluster with the left and another with the right, then widen the gap between them.
1176 | The separated network partitions are reconnecting. | Hold two separated clusters, then trace one connecting route across the gap and bring them slightly closer.
1177 | The replicas will become consistent after updates propagate. | Hold palms at different values, make two small alternating adjustments, then align them at equal height.
1178 | Let us read from enough replicas to reach a quorum. | Mark three node positions; present two together toward the center while leaving the third unchanged.
1179 | The system is switching to the backup service. | Hold the left as active node; lower it, then advance the right backup node into the active position.
1180 | We are dropping some nonessential load to protect the system. | Mark three incoming request paths; divert one outward before the shared service point.

### Cloud Deployment Concepts
1181 | Let us deploy a service. | Hold the left palm as a package; move the right from it into a framed cloud-area position and hold.
1182 | Let us add more instances to scale out. | Hold one service marker at center; add two separated right-hand markers beside it at equal height.
1183 | Let us reduce the number of instances to scale in. | Hold three service positions; retract two side positions and keep one central service marker.
1184 | Let us update the service gradually, one group at a time. | Mark three service instances; shift each one forward in sequence while the others remain in place.
1185 | Let us switch traffic from the current version to the new version. | Hold left and right environments apart; retract the left and advance the right into the active center position.
1186 | Let us release the new version to a small group first. | Hold a main group with the left; move one small right-hand instance forward alone and pause.
1187 | This is the threshold that triggers autoscaling. | Hold the left as a threshold line; move the right load marker upward past it, then add one side instance.
1188 | These containers are isolated from each other. | Frame two separate compact boxes with the palms and maintain a clear gap between them.
1189 | The service can discover the location of another service. | Mark three service positions; move the right request toward one selected endpoint and hold.
1190 | The service is failing over to another cloud region. | Hold two distant region positions; lower the active one and move the backup region marker forward.

### Cybersecurity Defense Concepts
1191 | I detected suspicious traffic. | Trace three normal request paths; deviate the fourth right-hand path and circle its endpoint once.
1192 | I am blocking a malicious request. | Move the right toward a protected area; place the left vertical palm across the path and stop it.
1193 | I am quarantining a device. | Mark one device among three, then move its marker sideways into a separate framed area.
1194 | I am patching a vulnerability. | Hold the left as a vulnerable gap; move the right across that gap as a covering boundary and hold.
1195 | Let us rotate a secret. | Hold the left as the old credential; retract it while the right takes the same service position.
1196 | Let me inspect an alert. | Present the left palm as an alert; circle it once with the right and pause above the center.
1197 | I am escalating a security incident. | Hold the left as an alert; raise the right from that point to a higher response level.
1198 | I am containing an incident. | Frame the incident area with both palms, then reduce the frame size and hold its boundaries.
1199 | We are recovering from an incident. | Hold a contained area with the left; trace the right outward along a controlled return-to-service path.
1200 | Let us close a security incident. | Bring both palms around the incident area, make one small synchronized downward beat, then return to neutral.

### Authentication & Authorization Concepts
1201 | Let me present a login challenge. | Hold the left palm vertically as a gate; raise the right palm before it and pause.
1202 | Here is the credential check. | Present the right palm as a credential; pass it beside the left gate and stop for verification.
1203 | Here is the successful authentication. | Move the right credential marker to the gate, then lower the left boundary and advance the right.
1204 | Here is the failed authentication. | Move the right toward the left gate; keep the gate fixed and retract the right after a brief hold.
1205 | Here is the multi-factor authentication. | Mark two separate credential positions with both hands, then advance them together toward one gate.
1206 | I am granting a permission. | Hold the left as a boundary; open a small passage and move the right through it.
1207 | I am denying a permission. | Hold the left as a boundary; move the right toward it and stop with a firm visible gap.
1208 | Here is the role-based access. | Mark three role levels with the right hand; connect one role to one permitted resource marked by the left.
1209 | Here is the least privilege. | Frame a broad access area, then narrow it to one smaller permitted region and hold.
1210 | I am revoking a session. | Hold the right as an active session; move it away from the protected area and return both hands to neutral.

### Machine Learning Lifecycle
1211 | Let us collect training samples. | Mark four sample positions with the right hand and gather them into one framed dataset area.
1212 | Let us split train and validation data. | Frame one dataset area, divide it into a larger left region and smaller right region, then hold.
1213 | Let us run model training cycle. | Hold the left as data; move the right through three repeated passes toward a model position at center.
1214 | Let us validate a model. | Hold the right as the trained model; present the left dataset beside it and make two comparison beats.
1215 | Let us tune a hyperparameter. | Hold the left as the model; move the right through three small level positions and settle on the middle.
1216 | I detected data leakage. | Hold the left palm as training data; trace a right-hand path from a protected evaluation area back into the training area.
1217 | I detected label noise. | Mark four sample positions; offset one right-hand label marker from its sample and circle that mismatch once.
1218 | Let us deploy a model. | Hold the left as the trained model; move the right from it toward a separated service area and hold.
1219 | Let us monitor model drift. | Hold the left as baseline; move the right gradually sideways across three checkpoints to show divergence.
1220 | Let us retrain after drift. | Show the drifted right position, return it to a data area, then trace a new training path toward center.

### Neural Network Concepts
1221 | Here is the input layer. | Mark three vertical input positions with the right hand while the left frames their shared side.
1222 | Here is the hidden layer. | Hold input positions on the left; mark a second compact column of internal nodes at center.
1223 | Here is the output layer. | Mark a final small column on the right and trace short links from the hidden layer toward it.
1224 | Here is the forward propagation. | Trace a right-hand path from left input through center to right output using three connected segments.
1225 | Here is the backpropagation. | Trace a return path from output toward input using the left hand along a lower separated route.
1226 | Here is the weight adjustment. | Hold two connected node positions; shift the right-hand connection path slightly and pause.
1227 | Here is the activation threshold. | Hold the left as a threshold line; move the right signal upward past it and make one forward output beat.
1228 | Here is the dropout. | Mark four hidden nodes, retract one marker to neutral, and keep the remaining three visible.
1229 | Here is the attention weighting. | Mark three input positions; bring one selected position slightly forward while the others stay back.
1230 | Here is the residual connection. | Trace a normal multi-step path, then add one curved bypass path from the first point to the final point.

### Computer Vision Concepts
1231 | Let me mark an image region. | Frame a compact rectangle with both palms at chest height and hold its borders steady.
1232 | Here is the bounding box adjustment. | Frame a rectangle, move only the right boundary inward, then the upper boundary downward slightly.
1233 | Here is the object detection. | Frame a broad scene with both hands, then outline one smaller target region inside it with the right.
1234 | Here is the object tracking. | Hold the left as the scene; move the right target marker through three connected positions across the scene.
1235 | Here is the image segmentation. | Frame one object region with the right hand, then trace a neighboring region boundary with the left.
1236 | Here is the edge detection. | Trace a smooth object outline with the right hand, then retrace only one sharp boundary segment more firmly.
1237 | Here is the depth ordering. | Hold the left palm in a rear plane and the right in a nearer plane, preserving visible separation.
1238 | Here is the stereo correspondence. | Mark one feature with each hand in parallel image planes, then align them to the same height.
1239 | Here is the optical flow. | Mark three nearby feature points, then move each slightly in the same direction with short sequential arcs.
1240 | Here is the occlusion. | Hold the left as a background object; slide the right palm partly in front of it, leaving one edge visible.

### Natural Language Processing Concepts
1241 | Let us tokenize a sentence. | Trace one horizontal sentence line with the right hand, then mark four separated token positions along it.
1242 | Let us merge subword tokens. | Mark two adjacent token positions and bring them closer into one combined central unit.
1243 | Here is the word embedding. | Hold one word marker with the right, move it into a compact semantic cluster framed by the left.
1244 | Here is the semantic similarity. | Present two word markers at separate positions, then move them closer while preserving a small gap.
1245 | Here is the semantic difference. | Present two nearby word markers, then separate them to opposite sides of the workspace.
1246 | Here is the context window. | Mark a central token and frame three neighboring token positions around it with both hands.
1247 | Here is the next-token prediction. | Mark three past tokens left to right, then place the right palm at one new future token position.
1248 | Here is the attention across words. | Mark four word positions; trace two cross-links from one selected word to two others.
1249 | Here is the sentiment shift. | Hold the left as a neutral baseline; move the right from below it to above it across one phrase span.
1250 | Here is the text classification. | Frame one text item, then move it with the right to one of three category positions.

### Speech & Audio Processing Concepts
1251 | Here is the audio waveform. | Trace a horizontal path with the right hand containing three small rising and falling wave pulses.
1252 | Here is the loudness increase. | Move the right through three progressively larger vertical pulse amplitudes while the left marks the baseline.
1253 | Here is the loudness decrease. | Move the right through three progressively smaller pulse amplitudes and settle on the baseline.
1254 | Here is the pitch rise. | Trace a smooth right-hand line that rises gradually from left to right across the chest-level space.
1255 | Here is the pitch fall. | Trace a smooth right-hand line that descends gradually from left to right.
1256 | Here is the noise removal. | Trace an uneven waveform, then retrace a smoother simplified waveform beside it.
1257 | Here is the voice activity region. | Trace a timeline with the left; bracket one active interval above it using both palms.
1258 | Here is the audio segmentation. | Trace one continuous line, then mark three clear boundaries along it with short vertical right-hand beats.
1259 | Here is the echo delay. | Make one right-hand forward pulse, pause, then repeat a smaller left-hand pulse at the same direction.
1260 | Here is the channel mix. | Hold two signal paths apart, guide both toward one shared output region, and maintain a small separation.

### Interface Interaction Concepts
1261 | Let me open a menu panel. | Hold the left palm as a screen; move the right from its side edge inward to frame a narrow panel.
1262 | Let us close a menu panel. | Frame a narrow side panel with the right hand, then slide it outward beyond the left screen boundary.
1263 | Let me expand a panel. | Frame a compact area with both palms and widen it horizontally in one smooth motion.
1264 | Let me collapse a panel. | Frame a wide area, bring the side boundaries closer, and hold the compact result.
1265 | Let me move a card. | Frame a small card with both hands, shift the intact frame sideways, and hold at the new position.
1266 | Let me resize a card. | Frame a small card, widen both side boundaries and raise the top boundary slightly.
1267 | Let me reorder two cards. | Mark two card positions, move the left one rightward and the right one leftward on separate depth planes.
1268 | Let me pin an item. | Mark one item with the left; draw the right hand to its upper right corner and make one small downward pin beat.
1269 | Let me unpin an item. | Mark a pinned item with the left; lift the right hand from its corner and draw it away smoothly.
1270 | Let me filter the view. | Frame a view with both palms, then sweep the right hand across to show filtered items only.

### Data Visualization Concepts
1271 | Here is a bar chart. | Mark four vertical bars of increasing height with the right hand while the left marks the horizontal axis.
1272 | Here is a line chart. | Trace a continuous rising and falling line with the right hand over a horizontal axis marked by the left.
1273 | Here is a pie chart. | Trace a horizontal circle with the right hand, then mark two radial slice lines from its center.
1274 | Here is a scatter plot. | Mark five scattered points at different height and depth positions with the right hand.
1275 | Here is a heatmap. | Frame a rectangular grid with both hands, then indicate three high-intensity cell areas with right-hand beats.
1276 | Here is a histogram distribution. | Trace a smooth distribution curve with the right hand over five vertical bar columns marked by the left.
1277 | Here is an area chart. | Trace an upper boundary line with the right hand and sweep downward to fill the area above the left baseline.
1278 | Here is a treemap layout. | Outline a main rectangle with both hands, then divide it into three nested sub-rectangles with the right.
1279 | Here is a radar chart. | Trace a star-like pentagon layout with the right hand, marking five radial axis endpoints from center.
1280 | Here is a waterfall chart. | Mark an initial bar with the left, then trace three rising and falling step bars with the right.

### User Onboarding Cues
1281 | Welcome to the platform. | Present both palms open and upward from chest height in a warm welcoming arc.
1282 | Let us set up your profile. | Frame a profile card area with the left hand and mark two entry fields with the right.
1283 | Let us guide you through the features. | Hold the left palm as a guide reference; move the right forward through three sequential feature checkpoints.
1284 | Step one of onboarding is complete. | Mark a progress bar with the left; move the right to the first marker with a downward confirmation beat.
1285 | Step two of onboarding is complete. | Advance the right marker along the progress bar to the second step and hold firmly.
1286 | You have finished onboarding. | Open both palms outward, then bring them together into a final appreciative hold.
1287 | Skip onboarding for now. | Present two palms forward, sweep them gently to the side, and settle in neutral pose.
1288 | Replay the onboarding tour. | Trace a small return circle with the right hand, then present the initial welcome pose.
1289 | Customise your preferences. | Frame a settings box with the left hand, and shift two toggle controls with the right.
1290 | Save your onboarding choices. | Present the selected preferences above the left palm, then make one firm confirmation beat with the right.

### System Health & Monitoring
1291 | All systems are operational. | Hold both palms level at chest height, then advance them slightly together with a calm settling pulse.
1292 | System status warning. | Present the left palm level; raise the right hand slightly higher with a small repeated pulse.
1293 | Critical system alert. | Raise both forearms to chest height, hold still, and make one sharp downward attention beat.
1294 | Monitoring server latency. | Trace a horizontal timeline with the left; mark three small latency peaks with the right.
1295 | Checking CPU load. | Hold the left as a 100% ceiling; move the right hand upward toward it and stop at 60%.
1296 | Checking memory usage. | Frame a memory container with both palms, then raise the right hand to mark current fill level.
1297 | Network bandwidth status. | Move both palms in parallel outward arcs to represent throughput volume.
1298 | Disk storage capacity. | Hold the left palm as storage volume; trace a right-hand boundary showing available space.
1299 | Service uptime status. | Trace a long uninterrupted horizontal line from left to right with the right hand.
1300 | Health check report summary. | Mark three green indicators with small right-hand beats, then present both palms flat and level.

### Robotics Interaction Cues
1301 | Robot state initialized. | Hold both forearms upright in a clean attention stance, then lower palms slightly into ready mode.
1302 | Robot arm home position. | Bring both forearms smoothly to the chest-level neutral presentation pose and freeze.
1303 | Robot trajectory planning. | Trace a smooth curved spatial path with the right hand while keeping the base stationary.
1304 | Robot obstacle avoidance. | Move the right hand forward, detect an obstacle with the left, and curve the right cleanly around it.
1305 | Robot gripper open. | Separate both palms outward horizontally to represent an opening end-effector.
1306 | Robot gripper close. | Move both palms inward toward each other, stopping with a clear gap to represent gripping.
1307 | Emergency stop cue. | Extend both vertical palms forward firmly at chest height and freeze immediately.
1308 | Robot manual control mode. | Hold the left palm as a control pad; move the right hand above it like a joystick indicator.
1309 | Robot autonomous mode. | Present both palms upward and let them move through two smooth pre-planned arcs without manual cues.
1310 | Robot task complete. | Finish motion with both hands centered, rotate palms upward, and lower forearms to waist height.

### Spatial Mapping & AR Cues
1311 | Scan physical environment. | Sweep the right palm in a slow 180-degree horizontal arc across the front workspace.
1312 | Detect planar surface. | Move the right palm downward to a horizontal plane at waist height and hold flat.
1313 | Place virtual object. | Hold an imaginary object with both hands and lower it gently onto the detected surface plane.
1314 | Anchor virtual object. | Lower the right palm onto a surface point and make one firm alignment beat to lock position.
1315 | Scale virtual object larger. | Move both palms outward diagonally away from center to expand the spatial box.
1316 | Scale virtual object smaller. | Bring both palms inward diagonally toward center to shrink the spatial box.
1317 | Rotate virtual object. | Hold palms on opposite sides of an imaginary box and rotate forearms in a circular steering arc.
1318 | Translate virtual object. | Frame an object with both hands and shift the intact frame horizontally to the side.
1319 | Hide spatial overlay. | Present the AR layer above the left palm, then sweep the right hand over it to clear the view.
1320 | Reveal spatial overlay. | Hold both palms closed near chest, then open them outward to reveal the spatial augmentation.

### Sensor Calibration & Diagnostic Cues
1321 | Start sensor calibration. | Hold the left forearm vertical as a reference mast; trace a right-hand circle around its midpoint.
1322 | Calibrate zero offset. | Hold the left palm level; bring the right hand to its exact center line and hold motionless.
1323 | Calibrate gain factor. | Hold two reference levels; shift the right marker between them to match the gain baseline.
1324 | Sensor noise detected. | Trace a small jittering line with the right hand over a baseline marked by the left.
1325 | Sensor signal filtered. | Trace a jittering line, then retrace a smooth clean line directly beside it.
1326 | Sensor out of range. | Hold upper boundary with the left; move the right signal past it and mark an out-of-bounds indicator.
1327 | Re-zero sensor baseline. | Move the right marker back to the left reference palm and execute one downward reset beat.
1328 | Diagnostic test passed. | Mark three test checkpoints with the right hand, then advance both palms forward in approval.
1329 | Diagnostic test failed. | Mark two test checkpoints, stop at the third with a vertical left palm, and retract.
1330 | Sensor calibration saved. | Hold calibrated values between palms, then bring palms together in a locking pulse.

### Presentation Opening & Closing
1331 | Welcome to today's presentation. | Open both arms outward from chest height in a warm, wide sweeping greeting.
1332 | Here is the main agenda. | Mark four descending chest-level rank positions with the right hand from top to bottom.
1333 | Let us move to the first topic. | Hold the left palm as agenda; move the right forward to topic one and hold.
1334 | Let us move to the next topic. | Shift the active right palm from topic one to topic two with a clean horizontal arc.
1335 | Let us summarize the presentation. | Bring both palms from separated positions to center, framing the complete topic space.
1336 | Thank you for your attention. | Bring both hands to chest, pause, then extend them forward gracefully with palms upward.
1337 | We are ready for your questions. | Present both palms open and upward toward the audience and hold motionless.
1338 | Concluding remarks. | Make two final summary beats with the right hand, then lower forearms smoothly to attention.
1339 | Transition to Q&A section. | Sweep the right palm from presentation screen direction toward audience direction.
1340 | End of presentation session. | Bring both palms together near chest, bow forearms slightly, and relax into ready stance.

### Q&A Management Cues
1341 | Who has the first question? | Present the right palm open toward audience and sweep slowly across the room.
1342 | I will answer that question now. | Point gently toward the questioner with the right palm, then bring the hand to chest line.
1343 | Please hold your question until the end. | Hold the left vertical palm as a stop signal, then point right hand toward the agenda.
1344 | That is an excellent question. | Make one appreciative right-hand beat at chest height with open palm facing upward.
1345 | Let me break down that answer. | Frame the question area with the left hand; mark three response points with the right.
1346 | Next question please. | Sweep the right palm from previous questioner to a new direction in the audience.
1347 | We have time for one more question. | Raise the right index-like presentation hand and hold for one clear beat.
1348 | Let us take that question offline. | Present the right palm, then guide it to a side follow-up space marked by the left.
1349 | Can you clarify your question? | Hold both palms upward in a non-threatening questioning pose and wait.
1350 | That completes our Q&A session. | Bring both palms to center, make one gentle downward beat, and lower forearms.

### Storytelling & Narrative Pacing
1351 | Once upon a time. | Trace a long opening arc with the right hand from back-workspace to front-center.
1352 | Setting the scene. | Outline a broad panoramic landscape plane with both hands at chest height.
1353 | Introducing the main character. | Hold the left palm as background; raise the right hand vertically as character entrance.
1354 | The plot thickens. | Move both palms in slowly tightening concentric arcs toward a central tension point.
1355 | The climax of the story. | Raise the right hand to a peak chest height, pause at the apex, and hold.
1356 | A sudden turning point. | Move right hand towards expected path, make a sharp right-angle shift, and hold.
1357 | The resolution unfolds. | Smoothly lower the peak right hand in a wide descending arc into a calm flat palm pose.
1358 | Looking back at the journey. | Curve the right hand backward toward past workspace while left holds present state.
1359 | Looking forward to the future. | Extend the right palm smoothly forward and slightly upward toward future horizon.
1360 | The final lesson of the story. | Frame the core moral above left palm with a sharp downward emphasis beat from right hand.

### Argumentation & Counterargument
1361 | Stating the main thesis. | Present the right palm firmly at chest height as the central thesis statement.
1362 | Supporting point number one. | Place the left palm beneath the thesis as a solid foundational support beat.
1363 | Supporting point number two. | Add a second right-hand support marker beside the left support base.
1364 | Introducing a counterargument. | Present left palm as thesis; place right palm at an opposing side position facing inward.
1365 | Refuting the counterargument. | Sweep right palm across the opposing argument position, clearing it away from thesis.
1366 | Weighing opposing claims. | Hold left and right palms at equal height, alternate small lifts, then favor the thesis side.
1367 | Finding common ground. | Bring opposing left and right palms toward center line until they align side-by-side.
1368 | Reinforcing the original thesis. | Bring right palm back to center thesis position with a strong downward confirmation beat.
1369 | Conceding a minor point. | Lower left palm slightly to acknowledge minor counterpoint while maintaining main right claim.
1370 | Final persuasive conclusion. | Present both palms open and forward, aligning all evidence into one unified direction.

### Risk & Safety Cues
1371 | Identifying a potential risk. | Hold left palm as process line; point right hand to a high-risk area above it with caution pulse.
1372 | Assessing risk severity. | Frame a risk area with left hand; raise right hand through low, medium, and high severity markers.
1373 | Implementing risk mitigation. | Place left palm over the risk area as a protective shield boundary.
1374 | Monitoring safety threshold. | Trace a safety ceiling with left palm; keep right operational line below that ceiling.
1375 | Safety boundary crossed. | Move right operational marker past the left safety boundary and signal alert pulse.
1376 | Fallback to safe mode. | Retract right hand from risky branch back to the left safe baseline path.
1377 | Risk level acceptable. | Lower right severity marker back to low level and confirm with flat palm hold.
1378 | Safety compliance verified. | Sweep right hand along the safety checklist with three firm check beats.
1379 | Emergency evacuation route. | Trace an urgent right-hand path out of danger zone toward designated safe exit.
1380 | Risk review completed. | Hold risk log above left palm, make one final review beat, and seal position.

### Compliance & Regulatory Cues
1381 | Checking regulatory standards. | Hold left palm as rulebook; trace right hand down its vertical margin line.
1382 | Audit standard passed. | Mark compliance check on left rulebook with a firm right-hand check beat.
1383 | Audit discrepancy found. | Circle a non-compliant region on left rulebook with right hand and hold gap.
1384 | Correcting compliance gap. | Move right hand into the discrepancy gap, realign edges, and lock in place.
1385 | Policy requirement met. | Frame policy boundary with left hand; place right palm securely within boundary.
1386 | Data privacy compliance. | Hold sensitive data above left palm; draw right boundary shield around it.
1387 | Security clearance verified. | Present security token with right hand, pass through left authorization gate.
1388 | Reporting compliance status. | Present summary report panel with left hand, add approval stamp beat with right.
1389 | Regulatory update required. | Shift old policy marker to the left, advance new updated policy marker to center.
1390 | Full compliance achieved. | Align both palms side by side, advance forward together, and hold in steady stance.

### Customer Support Cues
1391 | How can I assist you today? | Present both palms open, upward, and friendly at lower chest height.
1392 | Listening to your feedback. | Keep left palm relaxed near chest; tilt head forward slightly and present open right hand.
1393 | Resolving your issue. | Frame customer issue with left hand; bring right hand in to smooth out problem space.
1394 | Your issue has been resolved. | Move right hand cleanly across problem area and finish with a cheerful confirmation beat.
1395 | Escalating to support specialist. | Hold issue with left hand; raise right hand to higher specialist level.
1396 | Checking ticket status. | Hold ticket record above left palm; scan with right hand from top to bottom.
1397 | Following up on your request. | Move right hand forward along request line to check progress checkpoint.
1398 | Customer satisfaction confirmed. | Bring both palms together in appreciative gesture and hold with a smile-like posture.
1399 | Providing alternative solution. | Present primary option with left hand; offer secondary option with right hand.
1400 | Thank you for contacting support. | Present open palms forward, bow forearms gently, and transition to resting stance.

### Educational Feedback Cues
1401 | Great job on this exercise! | Raise right hand into an encouraging chest-level approval beat with open palm.
1402 | Good attempt, keep trying. | Present both palms upward gently, encouraging another forward attempt.
1403 | Let us correct this mistake. | Frame mistake location with left hand; redraw correct path with right hand.
1404 | Pay attention to this key step. | Tap left palm twice with right hand to highlight critical learning point.
1405 | You are making great progress. | Move right marker along learning timeline through two forward step markers.
1406 | Review this material again. | Trace a small loop with right hand returning to the lesson starting point.
1407 | Mastered this skill level. | Raise right hand past skill threshold line marked by left palm and hold.
1408 | Ready for the next challenge. | Present current level with left hand; point right hand forward to next level.
1409 | Remember this formula. | Frame formula region with both hands and hold boundaries steady for memory.
1410 | Lesson session complete. | Open both palms outward in summary, then settle hands into relaxed ready pose.

### Collaborative Problem Solving
1411 | Let us brainstorm ideas together. | Move both palms in active circular arcs around a shared central creative space.
1412 | Pooling our resources. | Move left and right hand markers from separate sides into one central shared pool.
1413 | Combining two complementary ideas. | Bring left idea and right idea together, meshing their paths into one unified concept.
1414 | Evaluating solution feasibility. | Hold solution option above left palm; test stability with short right-hand pulses.
1415 | Selecting the best team option. | Mark three group proposals; move the chosen right proposal to team center.
1416 | Division of team responsibilities. | Hold master task with left palm; distribute three sub-tasks to team positions.
1417 | Synchronizing team actions. | Move left and right hands forward in perfect parallel alignment and tempo.
1418 | Resolving team disagreement. | Separate conflicting hands, realign orientations, and bring back into consensus space.
1419 | Milestone reached by team. | Raise both palms together to mark a shared team milestone victory point.
1420 | Team solution finalized. | Lock both palms side by side over the completed project framework and hold.

### Creative Brainstorming Cues
1421 | Generating new creative ideas. | Trace expanding upward spirals with both hands from chest level into open space.
1422 | Thinking outside the box. | Frame a spatial box with left hand; move right hand far outside its boundaries.
1423 | Connecting distant concepts. | Mark concept A on far left and concept B on far right; trace long arc joining them.
1424 | Refining a creative concept. | Frame rough idea with left hand; trim and shape its edges gently with right hand.
1425 | Grouping similar ideas. | Cluster three right-hand idea markers together into one creative folder area.
1426 | Filtering out weak ideas. | Sweep right hand across idea board, keeping only the strong central candidate.
1427 | Building on an existing idea. | Place base idea on left palm; stack right-hand improvement layer directly on top.
1428 | Pivoting to a fresh perspective. | Hold original direction with left; turn right hand 90 degrees to explore new direction.
1429 | Spark of inspiration! | Make a quick upward opening pulse with right hand at chest height.
1430 | Finalizing creative concept. | Frame completed artwork/concept with both palms and present to audience.

### Time Management & Scheduling
1431 | Setting a project timeline. | Trace a long straight horizontal line from left to right at chest height.
1432 | Marking a deadline. | Trace timeline with right hand; place left vertical palm across it as hard deadline.
1433 | Task ahead of schedule. | Move right progress marker past current time marker set by left hand.
1434 | Task behind schedule. | Hold expected time marker with left; place actual progress marker behind it.
1435 | Adjusting schedule buffer. | Frame time interval with both palms and expand buffer zone by 20%.
1436 | Rescheduling meeting time. | Pick up schedule marker with right hand and shift to new position on timeline.
1437 | Time interval allocated. | Bracket start time and end time with both palms and hold space locked.
1438 | Recurring event schedule. | Trace three repeating circular arcs at regular intervals along timeline.
1439 | Time limit expiring. | Hold countdown line with left palm; bring right marker close to endpoint.
1440 | Schedule finalized and locked. | Press right hand onto calendar timeline with a solid confirmation beat.

### Resource Optimization
1441 | Analyzing resource consumption. | Hold resource pool with left palm; scan usage levels with right hand.
1442 | Eliminating resource waste. | Identify wasteful branch with right hand and sweep it away from main process.
1443 | Reallocating idle capacity. | Transfer idle capacity marker from low-priority left zone to high-priority right zone.
1444 | Maximizing efficiency. | Compress process path length with both palms while keeping output constant.
1445 | Balancing workload distribution. | Adjust left and right workload palms until their heights match perfectly.
1446 | Scaling resources on demand. | Expand resource boundary dynamically as incoming demand pulse arrives.
1447 | Conserving energy reserve. | Draw resource markers closer to chest and hold in low-power standby mode.
1448 | Recycling unused components. | Route discarded component from right exit back into left input stream.
1449 | Resource bottleneck cleared. | Open narrow bottleneck constraint, allowing smooth flow of right hand.
1450 | Optimal resource state achieved. | Present balanced resource matrix with both palms level and steady.

### Quality Assurance & Standards
1451 | Initiating quality inspection. | Hold product sample above left palm; inspect all faces with right hand.
1452 | Zero defect standard. | Sweep right hand smoothly over surface, verifying no defects present.
1453 | Tolerance margin check. | Check upper and lower tolerance limits with both palms around product height.
1454 | Stress testing component. | Apply controlled vertical test pulses with right hand onto product frame.
1455 | Quality threshold exceeded. | Raise product quality marker past standard baseline set by left palm.
1456 | Root cause defect analysis. | Trace defect back through three process steps to isolate faulty origin.
1457 | Standard operating procedure. | Follow SOP checklist with three precise sequential right-hand beats.
1458 | Continuous improvement loop. | Trace a continuous upward spiral showing iterative quality gains.
1459 | Quality certification granted. | Place official quality stamp beat onto product record with right hand.
1460 | QA audit successfully closed. | Fold audit file with both hands, make final approval pulse, and relax stance.

### Strategic Vision & Roadmap
1461 | Presenting the long-term vision. | Extend right arm forward and upward toward far horizon in inspiring arc.
1462 | Defining strategic goals. | Frame three strategic pillar positions across front workspace at chest level.
1463 | Phase one of roadmap. | Point to initial phase on left, establishing groundwork for strategy.
1464 | Phase two of roadmap. | Advance to intermediate phase at center, expanding scope and reach.
1465 | Phase three of roadmap. | Reach final mature phase on right, reaching peak strategic vision.
1466 | Aligning strategy with mission. | Align strategic arrow from right hand with central mission line from left.
1467 | Overcoming strategic hurdles. | Lift strategic path over high hurdle barrier and land cleanly beyond it.
1468 | Market expansion strategy. | Expand boundary paths outward in all directions from central home market.
1469 | Sustainable growth model. | Trace steady upward sloping line from left to right with no sharp drops.
1470 | Strategic roadmap approved. | Align all three strategic pillars, advance forward together, and hold firm.

### Change Management Cues
1471 | Preparing for organizational change. | Hold current state with left hand; indicate upcoming transition with right.
1472 | Communicating reason for change. | Present problem in current state, then show solution path leading to new state.
1473 | Guiding team through transition. | Support team through middle transition zone between old and new state.
1474 | Overcoming resistance to change. | Move gently against resistance barrier, easing it open with steady palm pressure.
1475 | Adopting new workflow. | Lower old workflow marker, elevate new workflow marker to active status.
1476 | Anchoring change in culture. | Press new workflow marker deep into team foundation with locking beat.
1477 | Feedback during transition. | Gather feedback inputs from side streams into central transition channel.
1478 | Celebrating change milestones. | Mark transition milestone victory with cheerful upward palm release.
1479 | Continuous adaptation. | Show flexible adjustment of workflow paths in response to environment shifts.
1480 | Change management complete. | Settle new organization state into calm, stable, operational baseline.

### Incident Command Cues
1481 | Incident reported and logged. | Log incoming alert marker with left hand, initiate command response with right.
1482 | Establishing incident command structure. | Set up command hierarchy levels with three clear right-hand rank markers.
1483 | Assessing incident impact zone. | Draw perimeter boundary around affected incident zone with right hand.
1484 | Deploying emergency response team. | Dispatch response team markers from command post to incident perimeter.
1485 | Containing incident spread. | Clamp containment boundary tight around incident zone with both palms.
1486 | Eliminating incident threat. | Sweep threat out of contained zone, verifying area is safe.
1487 | System recovery in progress. | Guide system components back to normal operational state step by step.
1488 | Incident post-mortem review. | Trace incident timeline backward from resolution to root cause trigger.
1489 | Preventative measures implemented. | Build reinforced safety barrier around vulnerable point to prevent recurrence.
1490 | Incident command stand-down. | Retract response teams, lower command post status, and restore normal ops.

### Summary & Call to Action Cues
1491 | Key takeaway number one. | Present first key takeaway with right hand at top chest level position.
1492 | Key takeaway number two. | Present second key takeaway at middle chest level position.
1493 | Key takeaway number three. | Present third key takeaway at lower chest level position.
1494 | Bringing all takeaways together. | Enclose all three takeaways in a single unified framing arc with both hands.
1495 | The final call to action. | Extend right palm forward forcefully with open palm, calling audience to act.
1496 | Join us in this mission! | Extend both open palms outward in a wide compelling invitation sweep.
1497 | Take the next step today. | Point to current position, then make one bold forward step marker with right hand.
1498 | Contact us for more details. | Present contact card space with left hand, point right hand to card details.
1499 | Thank you for joining us today! | Bring both hands to chest, pause with gratitude, then extend forward gracefully.
1500 | End of complete presentation catalogue. | Align both palms at center height, make one final firm confirmation beat, and drop to attention stance.
"""

def parse_catalogue_1001_1500():
    items = []
    current_category = "General Dialogue"
    
    for line in RAW_CATALOGUE_1001_1500.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("###"):
            current_category = line.replace("###", "").strip()
            continue
        
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0].isdigit():
            num = int(parts[0])
            phrase = parts[1]
            action = parts[2]
            gid = f"PDF_{num}"
            
            items.append({
                "gesture_id": gid,
                "number": num,
                "category": current_category,
                "intent": phrase,
                "spoken_response": phrase,
                "source_action": action,
                "motion_prompt": action,
                "kimodo_prompt": action,
                "source_catalogue": "1001-1500"
            })
    return items

def build_full_catalogue_500_1500():
    print("Building full gesture catalogue for IDs 500 to 1500...")
    
    # 1. Load ID 500 from gesture_library_250.json
    g500 = None
    if os.path.exists(JSON_250):
        with open(JSON_250, "r") as f:
            lib250 = json.load(f).get("gestures", [])
            for g in lib250:
                if g.get("number") == 500 or g.get("gesture_id") == "PDF_500":
                    g500 = {
                        "gesture_id": "PDF_500",
                        "number": 500,
                        "category": g.get("category", "Texture & Material Descriptions"),
                        "intent": g.get("intent", "Heavy Object"),
                        "spoken_response": "This object is very heavy.",
                        "source_action": g.get("motion_prompt", ""),
                        "motion_prompt": g.get("motion_prompt", ""),
                        "kimodo_prompt": g.get("kimodo_prompt", g.get("motion_prompt", "")),
                        "source_catalogue": "500-1500"
                    }
                    break
    
    if not g500:
        g500 = {
            "gesture_id": "PDF_500",
            "number": 500,
            "category": "Texture & Material Descriptions",
            "intent": "Heavy Object",
            "spoken_response": "This object is very heavy.",
            "source_action": "Hold both palms upward at waist height and raise them a short distance slowly, pausing halfway while keeping torso and feet steady.",
            "motion_prompt": "Hold both palms upward at waist height and raise them a short distance slowly, pausing halfway while keeping torso and feet steady.",
            "kimodo_prompt": "Hold both palms upward at waist height and raise them a short distance slowly, pausing halfway while keeping torso and feet steady.",
            "source_catalogue": "500-1500"
        }
        
    # 2. Load 501..1000 from gesture_library_501_1000.json
    g501_1000 = []
    if os.path.exists(JSON_501_1000):
        with open(JSON_501_1000, "r") as f:
            raw = json.load(f).get("gestures", [])
            for item in raw:
                # Add natural spoken_response if missing
                intent = item.get("intent", "")
                spoken = item.get("spoken_response", f"Let us focus on: {intent}.")
                
                g501_1000.append({
                    "gesture_id": item.get("gesture_id", f"PDF_{item.get('number')}"),
                    "number": item.get("number"),
                    "category": item.get("category", "Presentation"),
                    "intent": intent,
                    "spoken_response": spoken,
                    "source_action": item.get("motion_prompt", ""),
                    "motion_prompt": item.get("motion_prompt", ""),
                    "kimodo_prompt": item.get("kimodo_prompt", item.get("motion_prompt", "")),
                    "source_catalogue": "500-1500"
                })
                
    # 3. Parse 1001..1500 catalogue
    g1001_1500 = parse_catalogue_1001_1500()
    
    all_gestures = [g500] + g501_1000 + g1001_1500
    
    # Sort by gesture number
    all_gestures.sort(key=lambda x: x["number"])
    
    print(f"Total gestures compiled: {len(all_gestures)} (Range: {all_gestures[0]['number']} .. {all_gestures[-1]['number']})")
    
    payload = {
        "metadata": {
            "title": "Unitree G1 Gesture Catalogue (IDs 500-1500)",
            "total_count": len(all_gestures),
            "range": "500..1500",
            "robot": "Unitree G1 Humanoid",
            "joint_count": 29
        },
        "gestures": all_gestures
    }
    
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Saved catalogue JSON to {OUT_JSON}")
    return payload

if __name__ == "__main__":
    build_full_catalogue_500_1500()
