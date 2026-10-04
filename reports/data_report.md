# STATS19 data report

- **Source:** UK DfT STATS19 Road Safety Open Data - collision file(s)
- **Licence:** Open Government Licence v3.0 (attribution required: Contains public sector information licensed under the OGL v3.0). Confirm on the DfT page for your download date.
- **Years requested:** [2021, 2022, 2023, 2024, 2025]
- **Years used:** [2021, 2022, 2023, 2024, 2025]
- **Years missing:** none
- **Rows loaded:** 513,801   **Columns loaded:** 23
- **Duplicate full rows:** 0   **Duplicate collision_index:** 0

## Target

| Class | Count | % |
|---|---:|---:|
| Slight | 389,435 | 75.80 |
| Serious | 116,813 | 22.73 |
| Fatal | 7,553 | 1.47 |

Imbalance ratio (majority:minority) = 51.56. Expected frequency order Slight > Serious > Fatal.

## Missing values (loaded columns)

| Column | Missing | % |
|---|---:|---:|
| longitude | 53 | 0.01 |
| latitude | 53 | 0.01 |
| collision_index | 0 | 0.0 |
| collision_year | 0 | 0.0 |
| accident_severity | 0 | 0.0 |
| number_of_vehicles | 0 | 0.0 |
| date | 0 | 0.0 |
| day_of_week | 0 | 0.0 |
| first_road_class | 0 | 0.0 |
| road_type | 0 | 0.0 |
| speed_limit | 0 | 0.0 |
| junction_detail | 0 | 0.0 |
| junction_control | 0 | 0.0 |
| light_conditions | 0 | 0.0 |
| weather_conditions | 0 | 0.0 |
| road_surface_conditions | 0 | 0.0 |
| special_conditions_at_site | 0 | 0.0 |
| carriageway_hazards | 0 | 0.0 |
| urban_or_rural_area | 0 | 0.0 |
| trunk_road_flag | 0 | 0.0 |
| hour | 0 | 0.0 |
| target | 0 | 0.0 |
| severity | 0 | 0.0 |

## Unknown / special codes (% of rows)

| Column | % unknown |
|---|---:|
| road_type | 2.542 |
| junction_detail | 4.167 |
| junction_control | 44.591 |
| light_conditions | 0.007 |
| weather_conditions | 2.837 |
| road_surface_conditions | 1.995 |
| urban_or_rural_area | 0.011 |
| first_road_class | 0.001 |
| trunk_road_flag | 7.068 |
| special_conditions_at_site | 31.516 |
| carriageway_hazards | 1.925 |

## Suspicious values

- speed_limit_not_in_{20..70}: 3
- number_of_vehicles_lt_1: 0
- number_of_vehicles_gt_cap: 36
- unparseable_dates: 0
- hour_out_of_range_set_to_missing: 0
- coordinates_missing: 53
- coordinates_outside_uk_box: 0

## Outliers (IQR rule)

- speed_limit: 90964 (17.704%)
- number_of_vehicles: 10227 (1.99%)
- hour: 0 (0.0%)

Handling: number_of_vehicles capped at 10 (winsorised); speed_limit outside [20, 30, 40, 50, 60, 70] set to missing; not removing rows.

## Excluded features

| Feature | Reason |
|---|---|
| accident_severity | TARGET. Never a feature. |
| collision_severity | TARGET (DfT's name for accident_severity). |
| casualty_severity | LEAKAGE: severity of individual casualties; the collision severity is defined as the worst casualty severity. Casualty file is never loaded. |
| casualty_class | LEAKAGE/POST-EVENT: casualty-file field (describes who was hurt). Casualty file is never loaded. |
| casualty_type | LEAKAGE/POST-EVENT: casualty-file field. Casualty file is never loaded. |
| number_of_casualties | LEAKAGE: only known after the outcome, and mechanically linked to severity (a fatal collision has at least one casualty; more casualties -> more likely severe). |
| did_police_officer_attend_scene_of_accident | POST-EVENT: attendance is a response to the collision and is more likely for severe collisions, so it encodes the outcome. |
| collision_index | IDENTIFIER: no predictive meaning; kept only to de-duplicate and verify the split. |
| collision_ref_no | IDENTIFIER. |
| collision_year | Excluded to avoid memorising year-level drift (reporting changes, COVID-19 2020); kept only for analytics and temporal splitting. |
| date | Raw date replaced by engineered month / weekday / season. |
| time | Raw time replaced by engineered hour and derived flags. |
| day_of_week | Raw STATS19 code (1=Sunday) replaced by engineered weekday / is_weekend. |
| latitude | HIGH-CARDINALITY continuous geography. Not used as a model feature (location memorisation, weak generalisation); rounded copy kept for the historical map and grid aggregation only. |
| longitude | As latitude. |
| location_easting_osgr | As latitude (OSGR coordinates). |
| location_northing_osgr | As latitude (OSGR coordinates). |
| police_force | Administrative area with force-specific reporting practice; would let the model learn reporting differences rather than collision circumstances. (Move to ALLOWED_FEATURES to test it.) |
| local_authority_district | HIGH-CARDINALITY administrative identifier (~380 levels). |
| local_authority_ons_district | HIGH-CARDINALITY administrative identifier. |
| local_authority_highway | HIGH-CARDINALITY administrative identifier. |
| lsoa_of_accident_location | HIGH-CARDINALITY small-area identifier (~35k levels). |
| first_road_number | HIGH-CARDINALITY road identifier. |
| second_road_number | HIGH-CARDINALITY road identifier. |
| second_road_class | Structurally missing away from junctions and largely redundant with junction_detail. |
