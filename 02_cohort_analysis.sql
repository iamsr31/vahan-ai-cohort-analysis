-- ============================================================
-- VAHAN CASE STUDY — COHORT (LEAD-SOURCE) PERFORMANCE
-- Aggregation grain: lead_source (one row per cohort)
--
-- Why this grain?
--   The business question is "which lead sources convert best",
--   so the natural aggregate level is one row per lead_source.
--   Uploading at candidate/date grain would fragment small
--   cohorts and hide the pattern; rolling all the way up to a
--   single overall row would hide the differences we are asked
--   to find. lead_source is the exact level the case study
--   defines as a "cohort".
--
-- Primary ranking metric: FT_after_upload / Uploaded_Leads
--   (see report Page 2 for why this metric was chosen)
-- ============================================================

SELECT
    lead_source,

    -- volume
    SUM(uploaded_leads)                                            AS uploaded_leads,
    SUM(attempted)                                                 AS attempted,
    SUM(connected)                                                 AS connected,
    SUM(interested)                                                AS interested,
    SUM(ob_after_upload)                                           AS onboarded,
    SUM(ft_after_upload)                                           AS ft_after_upload,
    SUM(ft_after_first_attempt)                                    AS ft_after_first_attempt,

    -- funnel conversion rates
    ROUND(100.0 * SUM(attempted)  / NULLIF(SUM(uploaded_leads),0), 2) AS attempted_pct,
    ROUND(100.0 * SUM(connected)  / NULLIF(SUM(attempted),0),      2) AS attempt_to_connect_pct,
    ROUND(100.0 * SUM(interested) / NULLIF(SUM(connected),0),      2) AS connect_to_interested_pct,

    -- headline / ranking metric: overall FT conversion on total leads uploaded
    ROUND(100.0 * SUM(ft_after_upload) / NULLIF(SUM(uploaded_leads),0), 3) AS ft_rate_per_uploaded_pct,

    -- secondary metric: FT conversion on leads actually worked (attempted)
    ROUND(100.0 * SUM(ft_after_upload) / NULLIF(SUM(attempted),0), 3)      AS ft_rate_per_attempted_pct,

    -- operational efficiency: median time to first attempt
    ROUND(AVG(upload_to_first_attempt_p50_hrs), 1)                 AS avg_hrs_to_first_attempt

FROM raw_leads
GROUP BY lead_source
HAVING SUM(uploaded_leads) >= 20        -- drop micro cohorts (<20 leads) that are too small to rank reliably
ORDER BY ft_rate_per_uploaded_pct DESC;
