# agency_pymdp

* pymdp_agency_vBasic - uses older version of 

file name | configuration | issues and other notes
--- | ----- | -----
pymdp_agency_vBasic | <ul><li>2 state factors: context, and combined actions from agents</li><li>2 observational modalities: outcome, and action combination</li><li>uses older version of pymdp</li><li>bar graphs for visualisation</li></ul>
pymdp_agency_vBasic_v2 | <ul><li>2 state factors: context, and combined actions from agents</li><li>2 observational modalities: outcome, and action combination</li><li>uses newer version of pymdp (v 0.0.7)</li><li>grid colour graphs for visualisation</li></ul> | <ul><li>issue: actions are not chosen according to context - it always ends up at Action_compAction after t+2</li><li>issue: beliefs are not always in line with context</li></ul>
pymdp_agency_vBasic_v3 | <ul><li>2 state factors: context, and combined actions from agents</li><li>2 observational modalities: outcome, and action combination</li><li> added start states</li><li>uses newer version of pymdp (v 0.0.7)</li><li>grid colour graphs for visualisation</li></ul> | <ul><li>issue fixed from previous version: actions are chosen according to context</li><li>issue fixed from previous version: beliefs align with context - need to test out more rigorously</li><li>issue: the following error message shows up when running the kernel repeatedly "local variable 'observed_outcome' referenced before assignment"</ul>
pymdp_agency_vBasic_v4 | <ul><li>3 state factors: context, actions_agent1, and actions_agent2</li><li>3 observational modalities: outcome, actions_agent1, and actions_agent2</li><li>uses newer version of pymdp (v 0.0.7)</li><li>grid colour graphs for visualisation</li></ul> | This version works!

https://www.notion.so/pymdp-agency-task-model-d7a1c5b349c14313921e123f849209b7


