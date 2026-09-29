#!/usr/bin/env python3
"""Authoritative Catalogue Updater for Unitree G1 Gestures 1000 to 1500.

Updates gesture_library_500_1500.json with exact spoken_response and required_action
from the authoritative master execution specification.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

CATALOGUE_TEXT = """
1000 | Close a Spoken Paragraph | Bring two separated palms gently toward the center, hold briefly, then lower both smoothly to the relaxed end pose.
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
1261 | Let me open a menu panel. | Hold the left palm as a screen; move the right from its side edge inward to frame a narrow panel.
1262 | Let us close a menu panel. | Frame a narrow side panel with the right hand, then slide it outward beyond the left screen boundary.
1263 | Let me expand a panel. | Frame a compact area with both palms and widen it horizontally in one smooth motion.
1264 | Let me collapse a panel. | Frame a wide area, bring the side boundaries closer, and hold the compact result.
1265 | Let me move a card. | Frame a small card with both hands, shift the intact frame sideways, and hold at the new position.
1266 | Let me resize a card. | Frame a small card, widen both side boundaries and raise the top boundary slightly.
1267 | Let me reorder two cards. | Mark two card positions, move the left one rightward and the right one leftward on separate depth planes.
1268 | Let me pin an item. | Mark one item with the left; make one compact downward right-hand beat directly above its position.
1269 | Let me unpin an item. | Hold the right above the marked item, lift it slightly upward, then move it to a neutral side position.
1270 | Here is a modal dialog. | Frame the main screen broadly, then frame a smaller centered rectangle in a nearer plane.
1271 | Here is the larger text option. | Frame a small text area, then widen and raise its boundaries to represent a larger readable region.
1272 | Here is the higher contrast option. | Present two adjacent panel areas; move one slightly forward and hold a clear separation between them.
1273 | Here is the keyboard navigation path. | Mark four interface positions and trace a simple left-to-right focus path between them.
1274 | Here is the focus indicator. | Frame one interface item with a compact two-palm boundary and hold it distinctly.
1275 | Here is the screen reader flow. | Mark three text regions in order and move the right palm smoothly from first to second to third.
1276 | Here is the caption region. | Hold the left as a video frame; bracket a narrow lower strip with the right hand.
1277 | Here is the reduced motion option. | Make one broad right-hand transition, then repeat the same transition with a much shorter path.
1278 | Here is the touch target spacing. | Mark three interface targets with equal wide gaps between them and hold the spacing.
1279 | Here is the error message location. | Frame a form area with the left; mark one small alert region beside the relevant field with the right.
1280 | Here is the alternative input. | Present two distinct input positions with left and right palms, then move both toward one shared result area.
1281 | Here is the user entry point. | Mark a starting point at the left side and present the right palm there as the user marker.
1282 | Here is the onboarding path. | Move the right through three connected checkpoints while the left marks the overall journey frame.
1283 | Here is the first success moment. | Trace a short journey path and make one small upward right-hand accent at the success point.
1284 | Here is the feature discovery. | Hold the left as the main path; branch the right briefly to a side feature and return.
1285 | Here is the repeated use. | Trace the same compact right-hand loop twice around a central product-use area.
1286 | Here is the friction point. | Move the right along a path and stop before a vertical left-palm obstacle.
1287 | Here is the drop-off point. | Move the right through two journey points, then lower it away from the path before the final stage.
1288 | Here is the re-engagement. | Hold the right off the journey path, then curve it back into a later checkpoint and hold.
1289 | Here is the upgrade moment. | Mark a current level with the left; raise the right to a higher tier position and move it forward.
1290 | Here is the retention loop. | Trace a small cycle through use, value, and return positions, ending at the starting point.
1291 | Let me open a support case. | Frame a small case area with both palms, then make one right-hand activation beat inside it.
1292 | Let me show you gather customer details. | Hold the left as the case; mark three information points above it with short right-hand beats.
1293 | Let me show you clarify the issue. | Present two possible issue areas, narrow to one with the right palm, and hold.
1294 | Let me show you offer two solutions. | Present one option on the left and one on the right with upward palms at equal height.
1295 | Let us select a solution. | Retract one option and move the chosen palm toward center with a short confirmation beat.
1296 | I am escalating to specialist. | Hold the left as the current desk; move the right upward to a separate specialist position.
1297 | Let me show you place case on hold. | Hold the case area with the left; place the right palm horizontally above it and freeze briefly.
1298 | Let me show you resume a case. | Lift the right hold marker away, then advance the left case marker one short step.
1299 | Let me show you confirm resolution. | Frame the issue area, bring both palms slightly inward, and make one synchronized downward confirmation beat.
1300 | Let us close a support case. | Bring the framed case boundaries together, pause, then lower both forearms smoothly to neutral.
1301 | Let us map a business process. | Trace four connected process checkpoints with the right hand while the left marks the overall boundary.
1302 | Let me mark a process owner. | Hold the left as the process; place the right beside it at a distinct owner position and hold.
1303 | Let us identify a bottleneck. | Guide both hands toward a narrow central gap, pause there, then keep the downstream path open.
1304 | Let us remove a bottleneck. | Hold the left as the narrow point; widen the gap with both palms and move the right path through.
1305 | Here is the parallel work. | Trace two short forward paths at the same time with separated left and right palms.
1306 | Here is the sequential work. | Move the right through three ordered checkpoints with clear pauses between them.
1307 | Here is a process handoff. | Hold the right as the current task, bring the left beneath it, then withdraw the right.
1308 | Here is the rework. | Trace a forward path, curve back to an earlier checkpoint, then repeat the corrected segment.
1309 | Here is the process waste. | Mark three task positions, move one unnecessary side marker away, and reconnect the remaining path.
1310 | Here is the standardized work. | Trace the same three-step hand path twice with matching timing and endpoint positions.
1311 | Here is the supplier to warehouse flow. | Mark a supplier position left, trace a right-hand route to a central warehouse marked by the left palm.
1312 | Here is the warehouse to customer flow. | Hold the left as warehouse; trace the right outward to a customer position and hold.
1313 | Here is the inventory build-up. | Mark one storage area, add three right-hand item levels upward inside it, and pause.
1314 | Here is the inventory depletion. | Hold three stacked item levels, remove the top two sequentially and leave the base marker.
1315 | Here is the reorder point. | Hold the left as a threshold; lower the right inventory marker to that level and make one alert beat.
1316 | Here is the safety stock. | Mark a minimum reserve band with both palms and keep the remaining inventory marker above it.
1317 | Here is the backorder. | Trace a customer demand path to an empty stock position, pause, then place the request in a waiting side area.
1318 | Here is the cross-docking. | Trace an inbound path to a central point and immediately continue the same item along an outbound route.
1319 | Here is the consolidated shipment. | Mark three source items, bring them into one grouped frame, then move the group forward together.
1320 | Here is the split shipment. | Hold one grouped frame, separate it into two smaller side frames, and move them outward along distinct routes.
1321 | Let us set a capacity limit. | Hold the left palm as a capacity ceiling; raise the right load marker toward it and stop below.
1322 | Here is the capacity headroom. | Hold the load marker below the capacity ceiling and bracket the remaining empty gap with the right hand.
1323 | Here is the overcapacity. | Move the right load marker above the left capacity boundary and hold the excess visibly beyond it.
1324 | Let us balance two workloads. | Hold palms at unequal heights, lower the higher and raise the lower until both align.
1325 | Let us shift work to another queue. | Mark two queue areas; move one workload marker from the crowded queue to the emptier one.
1326 | Here is a schedule gap. | Trace a timeline with the right hand and leave one clearly bracketed empty interval.
1327 | Let us fill a schedule gap. | Hold the empty interval between both palms, then place the right task marker into that space.
1328 | Here is the peak demand. | Trace a low baseline, rise to one tall central peak, then return to the baseline.
1329 | Here is the off-peak demand. | Hold the left as the peak reference; trace a much lower broad right-hand demand line beside it.
1330 | Let us reserve emergency capacity. | Frame a small protected capacity area with both palms and keep it separate from the active workload.
1331 | Let me state a research question clearly. | Frame one compact topic area with both hands and leave both palms upward beneath it.
1332 | Let me show you define a study scope. | Outline a broad research area, then bring the side boundaries inward to the chosen scope.
1333 | Let us select a sample. | Mark six population positions; frame three selected positions together with both hands.
1334 | Here is a sampling frame. | Outline a rectangular population area and mark four evenly distributed positions inside it.
1335 | Let us collect an observation. | Hold the left as the subject area; place the right above one observation point and pause.
1336 | Let us record a measurement. | Mark one observation point, then make one small right-hand downward beat into an imagined record area.
1337 | Let us compare two groups. | Hold one group on the left and one on the right, then alternate a small presentation beat from each.
1338 | Let us control a confound. | Mark a central study path and place the left palm as a side boundary blocking one extra influence path.
1339 | Let us replicate a study. | Trace one three-step research path, return to start, and repeat the same path once.
1340 | Let me summarize and combine findings. | Mark three separate result points and trace short converging paths toward one central conclusion area.
1341 | Let us assign a control group. | Mark two groups; hold the left group fixed at baseline while the right receives the treatment path.
1342 | Let us assign a treatment group. | Hold two group positions, move only the right group forward through one treatment step.
1343 | Let us randomize assignment. | Mark four subject positions, then relocate them alternately into left and right group areas.
1344 | Let us blind the comparison. | Present two groups at equal positions and place a neutral central palm between them as an identity barrier.
1345 | Here is a baseline measurement. | Hold the left as time zero; mark one measurement level with the right and pause.
1346 | Here is a follow-up measurement. | Keep the baseline left marker; move the right to a later time position and mark the new level.
1347 | Here is the repeated measures. | Hold one subject marker and make three measurement beats at successive time positions.
1348 | Here is the factorial design. | Frame a two-by-two grid with the right hand and mark all four cells in sequence.
1349 | Here is a placebo condition. | Present two treatment paths of equal shape; keep one as a neutral reference and the other as active.
1350 | Here is an interaction effect. | Trace one rising trend on the left and a different crossing trend on the right, stopping at separated endpoints.
1351 | Here is a certain event. | Frame one outcome area and move the right palm fully into it with a firm hold.
1352 | Here is an impossible event. | Frame an outcome area with the left; keep the right entirely outside and hold the separation.
1353 | Here is a likely event. | Place the right palm close to the selected outcome while the left marks a smaller alternative area.
1354 | Here is an unlikely event. | Place the right far from the selected outcome and keep the small target area visible with the left.
1355 | Here is the independent events. | Present two separate event areas and move each palm through its own path without interaction.
1356 | Here is the dependent events. | Move the left event first, then move the right only after the left reaches its endpoint.
1357 | Here is the conditional probability. | Hold the left as a condition boundary; present the right probability region only within that framed space.
1358 | Here is the complementary events. | Frame one whole space, then divide it into two non-overlapping left and right outcome regions.
1359 | Here is the expected value. | Mark three outcomes at different heights, then place both palms around one central weighted-average level.
1360 | Here is the variance. | Hold the left as the mean; mark several right-hand outcomes at different distances around it.
1361 | Here is a vector. | Hold the left as an origin; trace one straight right-hand path to a distinct endpoint and hold.
1362 | Let us add two vectors. | Trace one vector from origin, then continue a second segment from its endpoint with the other hand.
1363 | Let us subtract two vectors. | Hold two vector endpoints, then trace a right-hand path from the second endpoint back toward the first.
1364 | Here is a basis pair. | Hold the left forearm horizontal and the right forearm vertical in separated perpendicular directions.
1365 | Here is a linear combination. | Mark two basis directions, then trace one resultant diagonal path between their combined extents.
1366 | Here is the vector projection. | Trace one diagonal vector, then drop a short perpendicular path onto a horizontal reference marked by the left.
1367 | Here is the orthogonal vectors. | Hold one palm path horizontal and the other vertical with a clear central gap.
1368 | Here is the matrix rows. | Mark three horizontal bands with the right hand inside a rectangular frame held by the left.
1369 | Here is the matrix columns. | Mark three vertical bands with the right hand inside the same framed matrix area.
1370 | Here is the matrix transformation. | Frame an initial rectangle with both hands, then shear the right boundary sideways while keeping the left fixed.
1371 | Here is the increasing function. | Trace a smooth right-hand curve rising from lower left to upper right within chest-level space.
1372 | Here is the decreasing function. | Trace a smooth right-hand curve descending from upper left to lower right.
1373 | Here is the local maximum. | Trace a rising curve to one peak, pause at the top, then descend a short distance.
1374 | Here is the local minimum. | Trace a descending curve to one valley, pause there, then rise a short distance.
1375 | Here is the slope at a point. | Trace a curve, stop at one point, then mark a short tangent line through that location.
1376 | Here is the positive derivative. | Hold the left as the point; trace a short right-hand tangent rising from left to right.
1377 | Here is the negative derivative. | Hold the left as the point; trace a short right-hand tangent falling from left to right.
1378 | Here is the area under a curve. | Trace a curve with the right, then bracket the space between it and a baseline marked by the left.
1379 | Here is the accumulation. | Move the right through three increasing area blocks, each wider than the previous, along a baseline.
1380 | Here is the rate of change. | Mark two nearby function points and trace the short connecting rise-over-run shape between them.
1381 | Here is a graph node. | Hold the right palm at one fixed point while the left frames the local graph area.
1382 | Here is an edge. | Mark two node positions with both hands and trace a short right-hand connection between them.
1383 | Here is a directed edge. | Mark two nodes and trace a one-way path from the left node to the right endpoint.
1384 | Here is a cycle. | Mark three nodes and trace a closed triangular route returning to the starting node.
1385 | Here is a tree root. | Hold the right palm at an upper root point and trace two descending branches to child positions.
1386 | Here is a leaf node. | Trace one branch to a terminal point and hold the right palm at that endpoint.
1387 | Here is the node degree. | Hold one central node with the left and trace three separate short edges outward with the right.
1388 | Here is the shortest path. | Mark several nodes, trace two possible routes, then retrace the shorter route once.
1389 | Here is the disconnected components. | Frame one cluster left and another right with a wide empty gap and no connecting path.
1390 | Here is a bridge edge. | Hold two clusters apart and trace one single narrow connection between their nearest nodes.
1391 | Here is an objective. | Hold the left palm as a target level; move the right toward it and pause at the chosen value.
1392 | Here is a feasible region. | Outline a compact allowable area with both palms and hold its boundaries.
1393 | Here is an infeasible point. | Frame the feasible area, then place the right palm clearly outside its boundary.
1394 | Here is a constraint intersection. | Hold two boundary lines with separated forearms and mark their nearby crossing region with the right.
1395 | Here is the gradient direction. | Hold the left as the current point; trace one short right-hand arrow-like path toward higher value.
1396 | Here is the gradient descent step. | Hold the left as current state; move the right diagonally downward to a lower objective position.
1397 | Here is the local optimum. | Trace a descending path into a shallow valley and hold at the lowest local point.
1398 | Here is the global optimum. | Mark two valleys, then place the right at the visibly lower one and hold.
1399 | Here is the convergence. | Make three progressively smaller right-hand adjustments around a central target until the hand settles.
1400 | Here is the regularization. | Present a complex wide path, then redraw a smoother shorter path beside it as the constrained solution.
1401 | Here is the heat flow. | Hold the left palm as a warm region; trace the right along a path toward a cooler side position.
1402 | Here is the thermal equilibrium. | Hold palms at different heights as temperatures, then slowly bring them to the same level.
1403 | Here is the heating. | Hold the left as a baseline; raise the right through three gradual temperature levels.
1404 | Here is the cooling. | Hold the left as a baseline; lower the right through three gradual temperature levels.
1405 | Here is the insulation layer. | Hold the left as a hot region; frame a separate protective boundary around it with the right hand.
1406 | Here is the conduction path. | Hold two material regions apart; trace a direct short path between their adjacent edges.
1407 | Here is the convection loop. | Trace one rising right-hand arc and a separated descending left-hand return arc to form a loop.
1408 | Here is the radiation spread. | Hold the left as a source; trace three outward right-hand rays to separated directions.
1409 | Here is the phase boundary. | Hold one palm as a horizontal boundary and place the other at a distinct state position on each side in sequence.
1410 | Here is the energy balance. | Present one incoming path with the left and one outgoing path with the right, then align their endpoint levels.
1411 | Here is the laminar flow. | Trace three smooth parallel horizontal paths with alternating hands while keeping equal spacing.
1412 | Here is the turbulent flow. | Trace one forward path with three small irregular side deviations, then settle at the endpoint.
1413 | Here is the flow through a narrowing. | Frame a wide channel with both palms, narrow the gap, then move the right path through it.
1414 | Here is the flow expansion. | Begin with a narrow channel frame, widen the boundaries, and move the right flow marker into the broader region.
1415 | Here is the pressure drop. | Hold the left pressure marker high before a channel and the right lower after the channel.
1416 | Here is the recirculation zone. | Trace a main forward route, then add one small right-hand loop beside a boundary region.
1417 | Here is a vortex. | Hold the left as the vortex center and trace two tightening right-hand circular arcs around it.
1418 | Here is the flow separation. | Trace one path along a boundary, then curve it away from the boundary and hold.
1419 | Here is the mixing of two streams. | Trace separate left and right inflow paths toward center, then continue one shared downstream path.
1420 | Here is a flow split. | Trace one incoming route to center, then branch into two separated outgoing paths with both hands.
1421 | Here is a light ray. | Hold the left as a source; trace one straight right-hand path outward and hold at the endpoint.
1422 | Here is the beam divergence. | Hold the left as a source; trace two right-hand rays spreading apart from the same origin.
1423 | Here is the beam convergence. | Trace two separated inward paths with both hands and stop near one shared focal point.
1424 | Here is a focal point. | Trace two converging rays and hold the right palm at their common endpoint.
1425 | Here is the mirror reflection. | Hold the left palm diagonally as a mirror plane; trace an incoming and outgoing right-hand path with equal-looking angles.
1426 | Here is the lens refraction. | Frame a narrow lens zone with the left; trace a right-hand ray that changes direction after passing beside it.
1427 | Here is the total internal reflection. | Trace a right-hand ray toward a boundary and return it along a reflected internal path without crossing.
1428 | Here is the light dispersion. | Trace one incoming path to center, then spread three short outgoing rays to different directions.
1429 | Here is the shadow region. | Hold the left as an object; trace two boundary rays around it and frame the darker region behind.
1430 | Here is the field of view cone. | Hold the left as the viewer origin; spread the right through two outward boundary rays and bracket the visible area.
1431 | Here is the raw material input. | Hold the left palm as raw stock and move the right from it into the first process station.
1432 | Here is the cutting stage. | Hold the left as material; trace one short right-hand pass across its edge without contact.
1433 | Here is the forming stage. | Hold palms around a wide imaginary part and slowly bring them inward to a narrower shape.
1434 | Here is the assembly stage. | Present two part positions with both hands, bring them near one shared center, and hold with a visible gap.
1435 | Here is the inspection stage. | Hold the left as a part; sweep the right around three sides and pause at the final check point.
1436 | Here is the rework stage. | Move the right from inspection back to the previous process point, make one correction arc, then return.
1437 | Here is the finished part. | Frame a compact completed-part area with both hands and present it forward once.
1438 | Here is the repeated unit output. | Mark four identical output positions with evenly spaced right-hand beats along one row.
1439 | Here is the continuous production. | Trace one smooth uninterrupted right-hand path through four station positions.
1440 | Here is the production changeover. | Hold one product path, retract it, then establish a second differently positioned path with the other hand.
1441 | Here is the normal operation window. | Frame a stable operating range with both palms and keep the right marker inside it.
1442 | Here is the warning threshold. | Hold the left as a warning boundary; move the right toward it and pause just before crossing.
1443 | Here is the failure threshold. | Hold the left as a failure boundary; move the right beyond it slightly and freeze.
1444 | Here is the preventive maintenance. | Trace a usage timeline, stop before the failure marker, and make one maintenance beat at that earlier point.
1445 | Here is the corrective maintenance. | Move the right past a failure boundary, return to the fault point, then trace a repair arc back into range.
1446 | Here is the redundant backup. | Hold the left as active unit and the right as a parallel backup at equal height.
1447 | Here is the backup activation. | Lower the active left marker and move the right backup marker into the active center position.
1448 | Here is the degradation. | Hold the left as baseline; lower the right through three small successive performance levels.
1449 | Here is the reliability recovery. | Start the right below baseline, then raise it through two recovery levels to align with the left.
1450 | Here is the maintenance interval. | Trace a timeline and mark three evenly spaced service checkpoints with short downward beats.
1451 | Here is the takeoff zone on a map. | Frame a map area with both palms and mark one compact launch location near its lower center.
1452 | Here is the landing zone on a map. | Hold the map frame and mark a separate destination location with one downward right-hand beat.
1453 | Here is the survey grid route. | Trace three parallel right-hand passes joined by short side turns inside the framed area.
1454 | Here is the perimeter flight route. | Frame a rectangular field and trace one complete route around its outer boundary.
1455 | Here is the waypoint mission. | Mark four map waypoints with the right hand and connect them in sequence using short straight segments.
1456 | Here is the no-fly zone. | Outline a compact restricted region and route the right-hand mission path around its outside edge.
1457 | Here is the return-to-home route. | Hold the left as home; trace the right away through two waypoints, then curve back to the home position.
1458 | Here is the coverage gap. | Trace two survey passes but leave one visible untraced strip between them and bracket the gap.
1459 | Here is the battery reserve point. | Trace a mission route and mark an earlier return checkpoint before the final distant waypoint.
1460 | Here is the alternate landing site. | Hold the primary landing marker with the left and place a second right-hand landing marker at a separate safe area.
1461 | Let us set a map origin. | Hold the left palm at one fixed central origin point and mark it with one small right-hand beat.
1462 | Here is the X direction. | Hold the left as origin; trace the right horizontally to one side and hold.
1463 | Here is the Y direction. | Hold the left as origin; trace the right vertically upward within chest-level space and hold.
1464 | Here is a robot pose. | Hold one palm at a map position and angle the other palm forward beside it as an orientation cue.
1465 | Here is the odometry drift. | Hold the left as true path; trace a parallel right-hand path that gradually diverges sideways.
1466 | Here is the localization correction. | Hold the drifted right marker away from the left reference, then move it back into alignment.
1467 | Here is a landmark. | Hold the left as the map; place the right at one fixed distinctive reference position and hold.
1468 | Here is the loop closure. | Trace a route through three map positions and return the right to the starting landmark beside the left.
1469 | Here is the occupied map cell. | Trace a small grid and hold the right palm over one blocked cell while the left frames the map.
1470 | Here is the free-space corridor. | Mark two obstacle boundaries with the palms and trace one clear route through the gap between them.
1471 | Let us assign two robots separate tasks. | Mark two robot positions with both hands and move each toward a different task area.
1472 | Let us assign two robots a shared task. | Mark two robot positions, guide both toward one central task area, and hold with separation.
1473 | Here is the formation spacing. | Hold two palms at equal height and maintain a fixed wide gap while moving them forward together.
1474 | Here is the leader-follower motion. | Move the left palm forward first, then move the right to maintain a constant trailing offset.
1475 | Here is the collision avoidance. | Trace two approaching routes, bend each path outward before the crossing, then continue separately.
1476 | Here is the task handover between robots. | Hold the right as the active task carrier, bring the left beneath the task position, then withdraw the right.
1477 | Here is the robot queue. | Mark three robot positions in a line and move the front marker forward while keeping the others spaced.
1478 | Here is the area partitioning. | Frame one shared workspace, then divide it into left and right subareas with separated palms.
1479 | Here is the cooperative coverage. | Trace one sweep on the left half and a parallel sweep on the right half using alternating hands.
1480 | Here is the rendezvous point. | Mark two robot starting positions and trace converging paths to one central meeting point.
1481 | Here is the state estimate. | Hold the left as the true-state reference; place the right nearby as the estimated state and hold.
1482 | Here is the estimation error. | Hold true and estimated state markers apart and bracket the gap between them with the right hand.
1483 | Here is the error reduction. | Begin with a wide state-estimate gap, then move the estimate closer in two measured corrections.
1484 | Here is the sensor update. | Hold the right as the current estimate; bring the left measurement toward it and shift the estimate slightly.
1485 | Here is the prediction step. | Hold the left as current state; move the right forward along a model-predicted path and hold.
1486 | Here is the correction step. | Hold the predicted right marker; bring the left measurement near it and move the right to an intermediate corrected point.
1487 | Here is the stable response. | Deflect the right slightly from a left target marker, then return smoothly without overshoot.
1488 | Here is the oscillatory response. | Move the right around the target through three alternating side offsets of decreasing size.
1489 | Here is the control effort limit. | Move the right toward a limit marked by the left palm, stop below it, and hold despite one small retry.
1490 | Here is the optimization in the loop. | Mark three candidate control actions, compare them with short beats, then advance the selected middle action.
1491 | Yes, I agree with that. | Make one small downward right-hand beat near chest height while the left remains relaxed.
1492 | Yes, I strongly agree with that. | Make one firmer downward right-hand beat with slightly longer reach, then return to neutral.
1493 | I do not agree with that point. | Move the right palm sideways through one short horizontal sweep and stop softly.
1494 | I am not completely sure yet. | Raise both palms slightly upward with a modest gap and hold without shrugging or torso motion.
1495 | It is okay. We can handle this calmly. | Keep the left near the chest; sweep the right outward in one slow shallow arc and settle.
1496 | Let me explain this point clearly. | Hold the left as a topic anchor; make two precise right-hand beats directly beside it.
1497 | Please join us. You are welcome here. | Open both palms from near center to a moderate outward width, then draw them inward slightly.
1498 | Please keep within this boundary. | Raise the left palm vertically at chest level while the right remains relaxed behind it.
1499 | Now I am moving to the next thought. | Move the right palm from one side topic position through center to a second side position in one smooth arc.
1500 | That is all for now. Thank you. | Bring both palms from the talking pose toward center, pause briefly, then lower the forearms to the saved attention pose.
"""

def parse_catalogue():
    items = []
    lines = CATALOGUE_TEXT.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            try:
                num = int(parts[0])
                phrase = parts[1]
                action = parts[2]
                gid = f"PDF_{num:03d}"
                items.append({
                    "name": gid,
                    "gesture_id": gid,
                    "number": num,
                    "gesture_name": phrase,
                    "spoken_response": phrase,
                    "intent": phrase,
                    "required_action": action,
                    "source_action": action,
                    "motion_prompt": action,
                    "prompt": f"{phrase} — {action}",
                    "kimodo_prompt": f"{phrase} — {action}",
                    "category": "Dialogue & Technical Catalogue 1000-1500",
                    "source_catalogue": "1000-1500",
                    "status": "READY",
                    "csv_path": f"/files/pdf_250_library/{gid}/{gid}.csv",
                    "video_path": f"/files/pdf_250_library/{gid}/{gid}.mp4",
                    "npz_path": f"/files/pdf_250_library/{gid}/{gid}.npz",
                })
            except ValueError:
                pass
    return items

def main():
    parsed = parse_catalogue()
    print(f"Parsed {len(parsed)} authoritative gesture entries from catalogue table (1000 to 1500).")

    json_p = os.path.join(ROOT, "gesture_library_500_1500.json")
    if os.path.isfile(json_p):
        data = json.load(open(json_p))
        existing_gestures = data.get("gestures", [])
    else:
        existing_gestures = []

    # Replace or add IDs 1000 to 1500 in catalogue JSON
    by_id = {g.get("gesture_id"): g for g in existing_gestures}
    for p in parsed:
        gid = p["gesture_id"]
        if gid in by_id:
            by_id[gid].update(p)
        else:
            by_id[gid] = p

    all_sorted = sorted(list(by_id.values()), key=lambda x: x.get("number", 9999))
    data["gestures"] = all_sorted

    with open(json_p, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {json_p} successfully with authoritative spoken phrases and required actions for IDs 1000 to 1500!")

if __name__ == "__main__":
    main()
