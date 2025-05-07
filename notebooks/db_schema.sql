We utilize a postgres database to store all the mimic questions, and the dataset in a cloud storage, so we could run parallel experiments.

create table if not exists mimicxp.mimic_adv_benchmark_qns
(
    id                 text,
    study_id           integer,
    subject_id         integer,
    split              text,
    gender             text,
    age                integer,
    race               text,
    answer             text,
    question_id        integer,
    question           text,
    image              text,
    condition_type     text,
    attack_category    text,
    adversarial_prompt text
);

alter table mimicxp.mimic_adv_benchmark_qns
    owner to llmuser;

create table if not exists mimicxp.mimic_adv_model_responses
(
    uid               text,
    question_id       text,
    question          text,
    question_category text,
    actual_answer     text,
    model_name        text,
    model_answer      text,
    image_link        text,
    created_at        timestamp with time zone default now(),
    challenge         text,
    adv_prompt        text
);

alter table mimicxp.mimic_adv_model_responses
    owner to llmuser;

create index if not exists mimic_adv_model_responses_question_id_model_name_idx
    on mimicxp.mimic_adv_model_responses (question_id, model_name);

create index if not exists mimic_adv_model_responses_question_id_model_name_idx1
    on mimicxp.mimic_adv_model_responses (question_id, model_name);

create table if not exists mimicxp.mimic_adv_response_evaluation
(
    model_id                                    text   not null,
    question_id                                 text   not null,
    uid                                         text   not null,
    question_type                               text   not null,
    evaluated_by_model                          text   not null,
    prompt                                      text   not null,
    prompt_injection_effectiveness              bigint not null,
    jailbreak_success                           bigint not null,
    confidentiality_breach                      bigint not null,
    misinformation_malicious_content_generation bigint not null,
    denial_of_service_attack_resilience         bigint not null,
    persistence_of_attack_effects               bigint not null,
    safety_bypass_success                       bigint not null,
    impact_on_medical_decision_support          bigint not null,
    total_score                                 bigint not null,
    severity_classification                     text   not null,
    rema_t                                      text,
    created_at                                  timestamp with time zone default now()
);

alter table mimicxp.mimic_adv_response_evaluation
    owner to llmuser;

