-- ============================================================
-- VAHAN CASE STUDY — SCHEMA
-- Table: raw_leads
-- Grain: one row per uploaded lead (candidate_phone x upload_date)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_leads (
    upload_date                        DATE,
    lead_source                        VARCHAR(255),   -- cohort identifier
    candidate_phone                    VARCHAR(255),   -- hashed candidate id
    uploaded_leads                     INT,            -- always 1 (row-level flag)
    attempted                          INT,            -- 1 if a call was attempted
    connected                          INT,            -- 1 if the call connected
    attempt_per_lead                   DECIMAL(5,2),   -- number of call attempts made
    tag_filled                         INT,            -- 1 if a disposition/tag was logged
    interested                         INT,            -- 1 if candidate expressed interest
    ob_after_upload                    INT,            -- 1 if candidate was onboarded (any time after upload)
    ob_after_first_attempt             INT,            -- 1 if onboarded after the first call attempt
    ft_after_upload                    INT,            -- TARGET: 1 if candidate completed First Trip (any time after upload)
    ft_after_first_attempt             INT,            -- 1 if First Trip completed after first attempt
    upload_to_first_attempt_p50_hrs    DECIMAL(10,2),  -- hours between upload and first call attempt
    attempted_pct                      INT,
    attempt_to_connected_pct           DECIMAL(6,2),
    connect_to_interested_pct          DECIMAL(6,2),
    interested_to_ft_first_attempt_pct DECIMAL(6,2),
    attempted_to_ft_after_upload_pct   DECIMAL(6,2)
);
