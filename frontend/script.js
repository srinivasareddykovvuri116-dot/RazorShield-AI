/* ============================================================
   RazorShield AI
   Frontend JavaScript
   ============================================================ */


/* ============================================================
   API CONFIGURATION
   ============================================================ */

const API_URL = "http://127.0.0.1:8000";


/* ============================================================
   GLOBAL STATE
   ============================================================ */

let demoTransactions = [];

let selectedDemoTransaction = null;


/*
    Demo history is kept separately from live history.

    Demo API:
        /api/v1/demo/transactions

    Live API:
        /api/v1/risk/history
*/

let demoHistoryDecisions = [];

let liveHistoryDecisions = [];


/* ============================================================
   DOM HELPERS
   ============================================================ */

function getElement(id) {
    return document.getElementById(id);
}


/* ============================================================
   HTML ESCAPE
   ============================================================ */

function escapeHtml(value) {

    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* ============================================================
   API HEALTH
   ============================================================ */

async function checkApiHealth() {

    const apiStatus =
        document.querySelector(".api-status");

    try {

        const response =
            await fetch(
                `${API_URL}/health`
            );

        if (!response.ok) {

            throw new Error(
                `HTTP ${response.status}`
            );

        }

        const data =
            await response.json();

        console.log(
            "RazorShield API health:",
            data
        );

        if (
            apiStatus &&
            data.status === "healthy"
        ) {

            apiStatus.innerHTML = `
                <span class="status-dot"></span>
                API Connected
            `;

        }

        else if (apiStatus) {

            apiStatus.innerHTML = `
                <span class="status-dot"></span>
                API Unavailable
            `;

        }

    }

    catch (error) {

        console.error(
            "API health check failed:",
            error
        );

        if (apiStatus) {

            apiStatus.innerHTML = `
                <span class="status-dot"></span>
                API Offline
            `;

        }

    }
}


/* ============================================================
   LOAD DEMO TRANSACTIONS
   ============================================================ */

async function loadDemoTransactions() {

    try {

        const response =
            await fetch(
                `${API_URL}/api/v1/demo/transactions`
            );

        if (!response.ok) {

            throw new Error(
                `Demo API returned HTTP ${response.status}`
            );

        }

        const data =
            await response.json();

        demoTransactions =
            Array.isArray(data.transactions)
                ? data.transactions
                : [];

        console.log(
            "Demo transactions loaded:",
            demoTransactions
        );

        return demoTransactions;

    }

    catch (error) {

        console.error(
            "Unable to load demo transactions:",
            error
        );

        demoTransactions = [];

        return [];

    }
}


/* ============================================================
   BUILD DEMO HISTORY
   ============================================================ */

/*
    The demo endpoint returns:

        {
            count: 8,
            transactions: [...]
        }

    Depending on backend version, a transaction may contain
    the decision information directly OR inside a transaction
    object.

    This function normalizes the response into the same shape
    used by the history renderer.
*/

function buildDemoHistory(transactions) {

    if (!Array.isArray(transactions)) {
        return [];
    }

    return transactions
        .map((item, index) => {

            if (!item) {
                return null;
            }

            const transaction =
                item.transaction &&
                typeof item.transaction === "object"
                    ? item.transaction
                    : item;

            const source =
                item.decision !== undefined ||
                item.risk_score !== undefined ||
                item.risk_level !== undefined
                    ? item
                    : transaction;

            const amount =
                Number(
                    source.amount ??
                    transaction.amount ??
                    0
                );

            const riskScore =
                Number(
                    source.risk_score ??
                    transaction.risk_score ??
                    0
                );

            const riskPercentage =
                Number(
                    source.risk_percentage ??
                    transaction.risk_percentage ??
                    (riskScore * 100)
                );

            const decision =
                String(
                    source.decision ??
                    transaction.decision ??
                    "UNKNOWN"
                ).toUpperCase();

            const riskLevel =
                String(
                    source.risk_level ??
                    transaction.risk_level ??
                    getRiskLevelFromDecision(
                        decision,
                        riskPercentage
                    )
                ).toUpperCase();

            const reasons =
                Array.isArray(source.reasons)
                    ? source.reasons
                    : (
                        Array.isArray(transaction.reasons)
                            ? transaction.reasons
                            : []
                    );

            const signals =
                Number(
                    source.signals ??
                    transaction.signals ??
                    reasons.length ??
                    0
                );

            const timestamp =
                source.timestamp ??
                transaction.timestamp ??
                transaction.created_at ??
                item.timestamp ??
                null;

            return {

                ...transaction,

                ...source,

                amount,

                risk_score: riskScore,

                risk_percentage:
                    Number.isFinite(riskPercentage)
                        ? riskPercentage
                        : 0,

                risk_level:
                    riskLevel,

                decision,

                signals,

                reasons,

                timestamp

            };

        })
        .filter(Boolean);

}


/* ============================================================
   FALLBACK RISK LEVEL
   ============================================================ */

function getRiskLevelFromDecision(
    decision,
    riskPercentage
) {

    if (decision === "BLOCK") {
        return "CRITICAL";
    }

    if (decision === "REVIEW") {
        return "HIGH";
    }

    if (
        Number.isFinite(riskPercentage) &&
        riskPercentage >= 25
    ) {
        return "HIGH";
    }

    return "LOW";
}


/* ============================================================
   FIND DEMO TRANSACTION
   ============================================================ */

function findDemoTransaction(scenario) {

    return demoTransactions.find(
        item =>
            item.scenario === scenario
    );
}


/* ============================================================
   SET FORM VALUE
   ============================================================ */

function setValue(id, value) {

    const element =
        getElement(id);

    if (!element) {
        return;
    }

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(value)
    ) {

        return;

    }

    element.value = value;
}


/* ============================================================
   GET NUMERIC VALUE
   ============================================================ */

function getNumber(
    id,
    fallback = 0
) {

    const element =
        getElement(id);

    if (!element) {
        return fallback;
    }

    const value =
        Number(element.value);

    if (!Number.isFinite(value)) {
        return fallback;
    }

    return value;
}


/* ============================================================
   GET SELECT VALUE
   ============================================================ */

function getSelectNumber(
    id,
    fallback = 0
) {

    const element =
        getElement(id);

    if (!element) {
        return fallback;
    }

    const value =
        Number(element.value);

    if (!Number.isFinite(value)) {
        return fallback;
    }

    return value;
}


/* ============================================================
   LOAD TRANSACTION INTO FORM
   ============================================================ */

function loadTransactionIntoForm(transaction) {

    if (!transaction) {
        return;
    }


    /* --------------------------------------------------------
       Core amount
    -------------------------------------------------------- */

    setValue(
        "amount",
        transaction.amount
    );


    /* --------------------------------------------------------
       Customer history
    -------------------------------------------------------- */

    let amountHistory =
        transaction.amount_vs_customer_history;


    /*
        Demo temporal rows use:

            amount_vs_historical_avg

        Older/manual requests may use:

            amount_vs_customer_history
    */

    if (
        amountHistory === undefined ||
        amountHistory === null
    ) {

        amountHistory =
            transaction.amount_vs_historical_avg;

    }


    if (
        amountHistory !== undefined &&
        amountHistory !== null
    ) {

        setValue(
            "amount-history",
            amountHistory
        );

    }


    /* --------------------------------------------------------
       Location
    -------------------------------------------------------- */

    setValue(
        "new-location",
        transaction.is_new_location ?? 0
    );

    setValue(
        "location-change",
        transaction.location_changed_from_previous ?? 0
    );


    /* --------------------------------------------------------
       IP
    -------------------------------------------------------- */

    let ipCustomers =
        transaction.ip_customer_count_before;


    if (
        ipCustomers === undefined
    ) {

        ipCustomers =
            transaction.ip_customer_count;

    }


    setValue(
        "ip-customers",
        ipCustomers ?? 0
    );


    /* --------------------------------------------------------
       Device
    -------------------------------------------------------- */

    let deviceCustomers =
        transaction.device_customer_count_before;


    if (
        deviceCustomers === undefined
    ) {

        deviceCustomers =
            transaction.device_customer_count;

    }


    setValue(
        "device-customers",
        deviceCustomers ?? 0
    );


    /* --------------------------------------------------------
       1 hour velocity
    -------------------------------------------------------- */

    let velocity1h =
        transaction.customer_txn_count_1h_before;


    if (
        velocity1h === undefined
    ) {

        velocity1h =
            transaction.customer_txn_count_1h;

    }


    setValue(
        "velocity-1h",
        velocity1h ?? 0
    );


    /* --------------------------------------------------------
       24 hour velocity
    -------------------------------------------------------- */

    let velocity24h =
        transaction.customer_txn_count_24h_before;


    if (
        velocity24h === undefined
    ) {

        velocity24h =
            transaction.customer_txn_count_24h;

    }


    setValue(
        "velocity-24h",
        velocity24h ?? 0
    );


    /* --------------------------------------------------------
       Hour
    -------------------------------------------------------- */

    setValue(
        "hour",
        transaction.hour_of_day ?? 12
    );


    /* --------------------------------------------------------
       High value
    -------------------------------------------------------- */

    setValue(
        "high-value",
        transaction.is_high_value ?? 0
    );


    /*
        Keep the complete demo transaction.
    */

    selectedDemoTransaction = {
        ...transaction
    };


    console.log(
        "Transaction loaded into form:",
        selectedDemoTransaction
    );
}


/* ============================================================
   DEMO BUTTONS
   ============================================================ */

function initializeDemoButtons() {

    const buttons =
        document.querySelectorAll(
            ".demo-button"
        );


    buttons.forEach(
        button => {

            button.addEventListener(
                "click",
                async function () {

                    const scenario =
                        button.dataset.scenario;


                    console.log(
                        "Selected demo scenario:",
                        scenario
                    );


                    /*
                        If demo data has not loaded yet,
                        load it now.
                    */

                    if (
                        demoTransactions.length === 0
                    ) {

                        await loadDemoTransactions();

                    }


                    const demo =
                        findDemoTransaction(
                            scenario
                        );


                    if (!demo) {

                        console.error(
                            "Demo transaction not found:",
                            scenario
                        );

                        showTemporaryError(
                            "Unable to load this demo transaction."
                        );

                        return;
                    }


                    /*
                        Store complete temporal transaction.
                    */

                    selectedDemoTransaction = {
                        ...demo.transaction
                    };


                    /*
                        Load visible fields into form.
                    */

                    loadTransactionIntoForm(
                        selectedDemoTransaction
                    );


                    /*
                        Highlight selected button.
                    */

                    buttons.forEach(
                        item =>
                            item.classList.remove(
                                "selected"
                            )
                    );


                    button.classList.add(
                        "selected"
                    );


                    /*
                        Reset previous result.
                    */

                    clearResult();


                    console.log(
                        "Selected demo transaction:",
                        selectedDemoTransaction
                    );

                }
            );

        }
    );
}


/* ============================================================
   BUILD MANUAL TRANSACTION
   ============================================================ */

function buildManualTransaction() {

    const amount =
        getNumber(
            "amount",
            0
        );


    const amountHistory =
        getNumber(
            "amount-history",
            0
        );


    const newLocation =
        getSelectNumber(
            "new-location",
            0
        );


    const locationChanged =
        getSelectNumber(
            "location-change",
            0
        );


    const ipCustomers =
        getNumber(
            "ip-customers",
            0
        );


    const deviceCustomers =
        getNumber(
            "device-customers",
            0
        );


    const velocity1h =
        getNumber(
            "velocity-1h",
            0
        );


    const velocity24h =
        getNumber(
            "velocity-24h",
            0
        );


    const hour =
        getNumber(
            "hour",
            12
        );


    const highValue =
        getSelectNumber(
            "high-value",
            0
        );


    return {

        amount: amount,

        amount_vs_customer_history:
            amountHistory,

        is_new_location:
            newLocation,

        location_changed_from_previous:
            locationChanged,

        ip_customer_count_before:
            ipCustomers,

        device_customer_count_before:
            deviceCustomers,

        device_transactions_before:
            0,

        ip_transactions_before:
            0,

        network_connections_before:
            0,

        customer_transactions_before:
            0,

        customer_txn_count_1h_before:
            velocity1h,

        customer_txn_count_24h_before:
            velocity24h,

        merchant_transactions_before:
            0,

        customer_amount_sum_before:
            0,

        customer_avg_amount_before:
            0,

        hour_of_day:
            hour,

        day_of_week:
            0,

        is_late_night:
            hour >= 0 && hour <= 5
                ? 1
                : 0,

        is_weekend:
            0,

        is_high_value:
            highValue,

        is_very_high_value:
            amount >= 50000
                ? 1
                : 0,

        device_other_customers_before:
            Math.max(
                deviceCustomers - 1,
                0
            ),

        ip_other_customers_before:
            Math.max(
                ipCustomers - 1,
                0
            ),

        device_ip_transactions_before:
            0

    };
}


/* ============================================================
   VALIDATE TRANSACTION
   ============================================================ */

function validateTransaction(transaction) {

    if (
        !transaction ||
        !Number.isFinite(
            Number(transaction.amount)
        ) ||
        Number(transaction.amount) <= 0
    ) {

        return {
            valid: false,
            message:
                "Please enter a valid transaction amount."
        };

    }


    if (
        transaction.amount_vs_customer_history !==
        undefined
    ) {

        const ratio =
            Number(
                transaction.amount_vs_customer_history
            );


        if (
            !Number.isFinite(ratio) ||
            ratio < 0
        ) {

            return {
                valid: false,
                message:
                    "Please enter a valid value for Amount vs Customer History."
            };

        }

    }


    return {
        valid: true
    };
}


/* ============================================================
   ANALYZE TRANSACTION
   ============================================================ */

async function analyzeTransaction(transaction) {

    const validation =
        validateTransaction(
            transaction
        );


    if (!validation.valid) {

        showTemporaryError(
            validation.message
        );

        return null;
    }


    setAnalyzingState(
        true
    );


    try {

        console.log(
            "Transaction sent to Risk Engine:",
            transaction
        );


        const response =
            await fetch(
                `${API_URL}/api/v1/risk/assess`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            transaction
                        )
                }
            );


        const data =
            await response.json();


        console.log(
            "Risk Engine HTTP status:",
            response.status
        );


        console.log(
            "Risk Engine response:",
            data
        );


        if (!response.ok) {

            let message =
                "Risk assessment failed.";


            if (
                data &&
                Array.isArray(
                    data.detail
                )
            ) {

                message =
                    data.detail
                        .map(
                            item =>
                                item.msg ||
                                "Validation error"
                        )
                        .join(
                            " "
                        );

            }

            else if (
                data &&
                typeof data.detail === "string"
            ) {

                message =
                    data.detail;

            }


            throw new Error(
                message
            );
        }


        displayResult(
            data
        );


        /*
            Refresh history after every successful
            live assessment.
        */

        await loadDecisionHistory();


        return data;

    }

    catch (error) {

        console.error(
            "Risk assessment error:",
            error
        );


        showTemporaryError(
            error.message ||
            "Unable to analyze transaction."
        );


        return null;

    }

    finally {

        setAnalyzingState(
            false
        );

    }
}


/* ============================================================
   FORM SUBMISSION
   ============================================================ */

function initializeRiskForm() {

    const form =
        getElement(
            "risk-form"
        );


    if (!form) {
        return;
    }


    form.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();


            /*
                If a demo transaction is selected,
                preserve the complete temporal vector.

                Otherwise create a manual transaction.
            */

            let transaction;


            if (
                selectedDemoTransaction
            ) {

                transaction = {
                    ...selectedDemoTransaction
                };


                /*
                    Update visible form fields.
                */

                transaction.amount =
                    getNumber(
                        "amount",
                        transaction.amount
                    );


                transaction.amount_vs_customer_history =
                    getNumber(
                        "amount-history",
                        transaction.amount_vs_historical_avg ??
                        0
                    );


                transaction.is_new_location =
                    getSelectNumber(
                        "new-location",
                        transaction.is_new_location ?? 0
                    );


                transaction.location_changed_from_previous =
                    getSelectNumber(
                        "location-change",
                        transaction.location_changed_from_previous ?? 0
                    );


                transaction.ip_customer_count_before =
                    getNumber(
                        "ip-customers",
                        transaction.ip_customer_count_before ??
                        0
                    );


                transaction.device_customer_count_before =
                    getNumber(
                        "device-customers",
                        transaction.device_customer_count_before ??
                        0
                    );


                transaction.customer_txn_count_1h_before =
                    getNumber(
                        "velocity-1h",
                        transaction.customer_txn_count_1h ??
                        0
                    );


                transaction.customer_txn_count_24h_before =
                    getNumber(
                        "velocity-24h",
                        transaction.customer_txn_count_24h ??
                        0
                    );


                transaction.hour_of_day =
                    getNumber(
                        "hour",
                        transaction.hour_of_day ??
                        12
                    );


                transaction.is_high_value =
                    getSelectNumber(
                        "high-value",
                        transaction.is_high_value ??
                        0
                    );


                /*
                    API expects:
                    amount_vs_customer_history
                */

                transaction.amount_vs_customer_history =
                    Number(
                        transaction.amount_vs_customer_history
                    );


                /*
                    Keep temporal dataset field synchronized.
                */

                if (
                    transaction.amount_vs_historical_avg !==
                    undefined
                ) {

                    transaction.amount_vs_historical_avg =
                        transaction.amount_vs_customer_history;

                }

            }

            else {

                transaction =
                    buildManualTransaction();

            }


            selectedDemoTransaction =
                transaction;


            await analyzeTransaction(
                transaction
            );

        }
    );
}


/* ============================================================
   ANALYZING UI STATE
   ============================================================ */

function setAnalyzingState(analyzing) {

    const button =
        getElement(
            "analyze-button"
        );


    const resultStatus =
        getElement(
            "result-status"
        );


    if (button) {

        button.disabled =
            analyzing;


        const text =
            button.querySelector(
                "span:first-child"
            );


        if (text) {

            text.textContent =
                analyzing
                    ? "Analyzing..."
                    : "Analyze Transaction";

        }

    }


    if (resultStatus) {

        resultStatus.textContent =
            analyzing
                ? "Analyzing"
                : "Analysis Complete";

    }
}


/* ============================================================
   DISPLAY RISK RESULT
   ============================================================ */

function displayResult(data) {

    if (!data) {
        return;
    }


    const scoreElement =
        getElement(
            "risk-score"
        );


    const levelElement =
        getElement(
            "risk-level"
        );


    const decisionElement =
        getElement(
            "decision"
        );


    const statusElement =
        getElement(
            "result-status"
        );


    const marker =
        getElement(
            "current-marker"
        );


    const circle =
        getElement(
            "risk-score-circle"
        );


    const percentage =
        Number(
            data.risk_percentage
        );


    const safePercentage =
        Number.isFinite(
            percentage
        )
            ? percentage
            : 0;


    /* --------------------------------------------------------
       Score
    -------------------------------------------------------- */

    if (scoreElement) {

        scoreElement.textContent =
            safePercentage.toFixed(2);

    }


    /* --------------------------------------------------------
       Level
    -------------------------------------------------------- */

    if (levelElement) {

        levelElement.textContent =
            data.risk_level ||
            "UNKNOWN";

    }


    /* --------------------------------------------------------
       Decision
    -------------------------------------------------------- */

    if (decisionElement) {

        decisionElement.textContent =
            data.decision ||
            "UNKNOWN";

    }


    /* --------------------------------------------------------
       Status
    -------------------------------------------------------- */

    if (statusElement) {

        statusElement.textContent =
            "Analysis Complete";

    }


    /* --------------------------------------------------------
       Risk CSS classes
    -------------------------------------------------------- */

    const riskClass =
        getRiskClass(
            data
        );


    if (circle) {

        circle.classList.remove(
            "allow",
            "review",
            "block",
            "low",
            "medium",
            "high",
            "critical"
        );


        circle.classList.add(
            riskClass
        );

    }


    if (levelElement) {

        levelElement.classList.remove(
            "allow",
            "review",
            "block",
            "low",
            "medium",
            "high",
            "critical"
        );


        levelElement.classList.add(
            riskClass
        );

    }


    if (decisionElement) {

        decisionElement.classList.remove(
            "allow",
            "review",
            "block",
            "low",
            "medium",
            "high",
            "critical"
        );


        decisionElement.classList.add(
            riskClass
        );

    }


    /* --------------------------------------------------------
       Risk marker
    -------------------------------------------------------- */

    if (marker) {

        const markerPercentage =
            Math.min(
                Math.max(
                    safePercentage,
                    0
                ),
                100
            );


        marker.style.left =
            `${markerPercentage}%`;

    }


    /* --------------------------------------------------------
       Reasons
    -------------------------------------------------------- */

    renderReasons(
        data.reasons || []
    );
}


/* ============================================================
   RISK CLASS
   ============================================================ */

function getRiskClass(data) {

    const decision =
        String(
            data.decision || ""
        ).toUpperCase();


    const level =
        String(
            data.risk_level || ""
        ).toUpperCase();


    if (
        decision === "BLOCK"
    ) {

        return "block";

    }


    if (
        decision === "REVIEW"
    ) {

        return "review";

    }


    if (
        decision === "ALLOW"
    ) {

        return "allow";

    }


    if (
        level === "CRITICAL"
    ) {

        return "critical";

    }


    if (
        level === "HIGH"
    ) {

        return "high";

    }


    if (
        level === "MEDIUM"
    ) {

        return "medium";

    }


    return "low";
}


/* ============================================================
   RENDER REASONS
   ============================================================ */

function renderReasons(reasons) {

    const container =
        getElement(
            "reasons"
        );


    if (!container) {
        return;
    }


    container.innerHTML =
        "";


    if (
        !Array.isArray(reasons) ||
        reasons.length === 0
    ) {

        container.innerHTML = `
            <div class="empty-state">
                No major behavioral risk signal detected.
            </div>
        `;

        return;
    }


    reasons.forEach(
        reason => {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "reason-item";


            item.innerHTML = `
                <span class="reason-icon">!</span>

                <span class="reason-text">
                    ${escapeHtml(reason)}
                </span>
            `;


            container.appendChild(
                item
            );

        }
    );
}


/* ============================================================
   CLEAR RESULT
   ============================================================ */

function clearResult() {

    const score =
        getElement(
            "risk-score"
        );


    const level =
        getElement(
            "risk-level"
        );


    const decision =
        getElement(
            "decision"
        );


    const status =
        getElement(
            "result-status"
        );


    const reasons =
        getElement(
            "reasons"
        );


    const marker =
        getElement(
            "current-marker"
        );


    if (score) {

        score.textContent =
            "—";

    }


    if (level) {

        level.textContent =
            "Awaiting transaction";

    }


    if (decision) {

        decision.textContent =
            "—";

    }


    if (status) {

        status.textContent =
            "Waiting";

    }


    if (marker) {

        marker.style.left =
            "0%";

    }


    if (reasons) {

        reasons.innerHTML = `
            <div class="empty-state">
                Analyze a transaction to
                see the AI explanation.
            </div>
        `;

    }
}


/* ============================================================
   TEMPORARY ERROR
   ============================================================ */

function showTemporaryError(message) {

    const status =
        getElement(
            "result-status"
        );


    const reasons =
        getElement(
            "reasons"
        );


    if (status) {

        status.textContent =
            "Error";

    }


    if (reasons) {

        reasons.innerHTML = `
            <div class="empty-state error-state">
                ${escapeHtml(message)}
            </div>
        `;

    }
}


/* ============================================================
   LOAD LIVE DECISION HISTORY
   ============================================================ */

async function loadDecisionHistory() {

    console.log("=== LOADING DECISION HISTORY === - script.js:1887");

    /*
     * ========================================================
     * DEMO HISTORY
     * ========================================================
     *
     * These are representative scenarios supplied by:
     *
     *     /api/v1/demo/transactions
     *
     * IMPORTANT:
     *
     * We DO NOT send these transactions back to
     * /api/v1/risk/assess here.
     *
     * Doing that would create duplicate LIVE history
     * entries every time the History page is opened.
     */

    if (demoTransactions.length === 0) {

        await loadDemoTransactions();

    }


    /*
     * These are the model-generated results already verified
     * for the eight representative demo scenarios.
     *
     * They correspond to the eight transactions returned by
     * /api/v1/demo/transactions in the same order.
     */

    demoHistoryDecisions = [

        {
            demo_id: "DEMO-SAFE-001",
            scenario: "SAFE",
            description:
                "Normal transaction from the processed temporal feature dataset.",

            timestamp: "Demo scenario",

            amount: 526.60,

            risk_score: 0.0001,

            risk_percentage: 0.01,

            risk_level: "LOW",

            decision: "ALLOW",

            signals: 1,

            reasons: [
                "No major behavioral risk signal detected."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-REVIEW-001",
            scenario: "REVIEW",
            description:
                "Genuine borderline fraud example from the validation dataset.",

            timestamp: "Demo scenario",

            amount: 3610.98,

            risk_score: 0.188,

            risk_percentage: 18.80,

            risk_level: "HIGH",

            decision: "REVIEW",

            signals: 3,

            reasons: [
                "IP address is associated with multiple customers.",
                "Device is associated with multiple customers.",
                "Transaction has multiple previous network connections."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-HIGH-VALUE-001",
            scenario: "HIGH_VALUE",
            description:
                "Real dataset transaction with significant amount deviation and high-value behavior.",

            timestamp: "Demo scenario",

            amount: 24512.08,

            risk_score: 0.9563,

            risk_percentage: 95.63,

            risk_level: "CRITICAL",

            decision: "BLOCK",

            signals: 2,

            reasons: [
                "Transaction amount is significantly above the customer's historical average.",
                "High-value transaction detected."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-NEW-LOCATION-001",
            scenario: "NEW_LOCATION",
            description:
                "Real dataset transaction originating from a previously unseen customer location.",

            timestamp: "Demo scenario",

            amount: 11370.71,

            risk_score: 0.9925,

            risk_percentage: 99.25,

            risk_level: "CRITICAL",

            decision: "BLOCK",

            signals: 2,

            reasons: [
                "Transaction originates from a new customer location.",
                "Transaction amount is significantly above customer history."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-LOCATION-CHANGE-001",
            scenario: "LOCATION_CHANGE",
            description:
                "Real dataset transaction where the customer location changed from the previous transaction.",

            timestamp: "Demo scenario",

            amount: 2621.51,

            risk_score: 0.9906,

            risk_percentage: 99.06,

            risk_level: "CRITICAL",

            decision: "BLOCK",

            signals: 3,

            reasons: [
                "Customer transaction location changed from the previous transaction.",
                "Transaction originates from a new customer location.",
                "Transaction behavior differs from customer history."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-HIGH-VELOCITY-001",
            scenario: "HIGH_VELOCITY",
            description:
                "Real dataset transaction showing elevated transaction velocity within one hour.",

            timestamp: "Demo scenario",

            amount: 131.85,

            risk_score: 0.0002,

            risk_percentage: 0.02,

            risk_level: "LOW",

            decision: "ALLOW",

            signals: 1,

            reasons: [
                "No major behavioral risk signal detected."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-NETWORK-RISK-001",
            scenario: "NETWORK_RISK",
            description:
                "Real dataset transaction with strong network and shared-entity relationships.",

            timestamp: "Demo scenario",

            amount: 11754.22,

            risk_score: 0.9048,

            risk_percentage: 90.48,

            risk_level: "CRITICAL",

            decision: "BLOCK",

            signals: 2,

            reasons: [
                "IP address is associated with multiple customers.",
                "Device is associated with multiple customers."
            ],

            _history_source: "DEMO"

        },


        {
            demo_id: "DEMO-BLOCK-001",
            scenario: "HIGH_RISK",
            description:
                "High-risk transaction with multiple strong fraud signals.",

            timestamp: "Demo scenario",

            amount: 45000.00,

            risk_score: 0.9926,

            risk_percentage: 99.26,

            risk_level: "CRITICAL",

            decision: "BLOCK",

            signals: 8,

            reasons: [
                "Transaction amount is more than 5x the customer's historical average.",
                "Transaction originates from a new customer location.",
                "Customer transaction location changed from the previous transaction.",
                "IP address is associated with multiple customers.",
                "Device is associated with multiple customers.",
                "Transaction has multiple previous network connections.",
                "Unusually high transaction velocity detected within one hour.",
                "High-value transaction detected."
            ],

            _history_source: "DEMO"

        }

    ];


    console.log(
        "Demo history loaded:",
        demoHistoryDecisions.length
    );


    /*
     * ========================================================
     * LIVE HISTORY
     * ========================================================
     *
     * Only read the live endpoint.
     *
     * NEVER call /risk/assess here.
     */

    try {

        const response =
            await fetch(
                `${API_URL}/api/v1/risk/history`
            );


        if (!response.ok) {

            throw new Error(
                `Decision history request failed: ${response.status}`
            );

        }


        const data =
            await response.json();


        liveHistoryDecisions =
            Array.isArray(data.decisions)
                ? data.decisions
                : [];


        console.log(
            "Live decisions loaded:",
            liveHistoryDecisions.length
        );


    }

    catch (error) {

        console.error(
            "Unable to load live decision history:",
            error
        );


        liveHistoryDecisions = [];

    }


    /*
     * ========================================================
     * COMBINE DEMO + LIVE
     * ========================================================
     */

    const combinedHistory = [

        ...demoHistoryDecisions,

        ...liveHistoryDecisions.map(
            decision => ({
                ...decision,
                _history_source: "LIVE"
            })
        )

    ];


    console.log(
        "Combined history:",
        combinedHistory.length
    );


    /*
     * ========================================================
     * RENDER
     * ========================================================
     */

    renderCombinedDecisionHistory(
        combinedHistory
    );

}


/* ============================================================
   COMBINE DEMO + LIVE HISTORY
   ============================================================ */

function combineHistoryDecisions(
    demoDecisions,
    liveDecisions
) {

    const demo =
        Array.isArray(demoDecisions)
            ? demoDecisions
            : [];


    const live =
        Array.isArray(liveDecisions)
            ? liveDecisions
            : [];


    /*
        Keep every demo scenario.

        Then append live decisions.

        We deliberately do NOT deduplicate live decisions
        against demo decisions by amount because a user may
        legitimately analyze the same amount.
    */

    return [
        ...demo.map(
            item => ({
                ...item,
                _history_source: "DEMO"
            })
        ),

        ...live.map(
            item => ({
                ...item,
                _history_source: "LIVE"
            })
        )
    ];

}


/* ============================================================
   RENDER COMBINED HISTORY
   ============================================================ */

function renderCombinedDecisionHistory(
    decisions
) {

    const safeDecisions =
        Array.isArray(decisions)
            ? decisions
            : [];


    const total =
        safeDecisions.length;


    const allowed =
        safeDecisions.filter(
            item =>
                String(
                    item.decision || ""
                ).toUpperCase() ===
                "ALLOW"
        ).length;


    const reviews =
        safeDecisions.filter(
            item =>
                String(
                    item.decision || ""
                ).toUpperCase() ===
                "REVIEW"
        ).length;


    const blocked =
        safeDecisions.filter(
            item =>
                String(
                    item.decision || ""
                ).toUpperCase() ===
                "BLOCK"
        ).length;


    updateHistorySummary(
        total,
        allowed,
        reviews,
        blocked
    );


    updateHistoryDemoLabel(
        "8 representative demo scenarios + live analyzed transactions."
    );


    renderHistoryRows(
        safeDecisions
    );

}


/* ============================================================
   RENDER DEMO HISTORY
   Compatibility function
   ============================================================ */

function renderDecisionHistory(data) {

    const decisions =
        Array.isArray(data.decisions)
            ? data.decisions
            : [];


    renderCombinedDecisionHistory(
        decisions
    );
}


/* ============================================================
   RENDER LIVE HISTORY
   Compatibility function
   ============================================================ */

function renderLiveDecisionHistory(data) {

    const decisions =
        Array.isArray(data.decisions)
            ? data.decisions
            : [];


    /*
        Keep the demo history visible while adding
        the live decisions.
    */

    liveHistoryDecisions =
        decisions;


    const combinedHistory =
        combineHistoryDecisions(
            demoHistoryDecisions,
            liveHistoryDecisions
        );


    renderCombinedDecisionHistory(
        combinedHistory
    );
}


/* ============================================================
   UPDATE HISTORY SUMMARY
   ============================================================ */

function updateHistorySummary(
    total,
    allowed,
    reviews,
    blocked
) {

    const totalElement =
        document.getElementById(
            "history-total"
        );


    const allowedElement =
        document.getElementById(
            "history-allowed"
        );


    const reviewElement =
        document.getElementById(
            "history-review"
        );


    const blockedElement =
        document.getElementById(
            "history-blocked"
        );


    if (totalElement) {

        totalElement.textContent =
            total;

    }


    if (allowedElement) {

        allowedElement.textContent =
            allowed;

    }


    if (reviewElement) {

        reviewElement.textContent =
            reviews;

    }


    if (blockedElement) {

        blockedElement.textContent =
            blocked;

    }


    /* --------------------------------------------------------
       Data attribute compatibility
    -------------------------------------------------------- */

    const totalData =
        document.querySelector(
            "[data-history-total]"
        );


    const allowedData =
        document.querySelector(
            "[data-history-allowed]"
        );


    const reviewData =
        document.querySelector(
            "[data-history-review]"
        );


    const blockedData =
        document.querySelector(
            "[data-history-blocked]"
        );


    if (totalData) {

        totalData.textContent =
            total;

    }


    if (allowedData) {

        allowedData.textContent =
            allowed;

    }


    if (reviewData) {

        reviewData.textContent =
            reviews;

    }


    if (blockedData) {

        blockedData.textContent =
            blocked;

    }

}


/* ============================================================
   HISTORY DATA LABEL
   ============================================================ */

function updateHistoryDemoLabel(description) {

    const label =
        document.getElementById(
            "history-data-label"
        );


    if (!label) {
        return;
    }


    label.textContent =
        "DEMO + LIVE DATA";


    label.title =
        description || "";

}


/* ============================================================
   FORMAT HISTORY TIMESTAMP
   ============================================================ */

function formatHistoryTimestamp(timestamp) {

    if (!timestamp) {

        return "—";

    }


    /*
        Some demo data already contains a formatted timestamp
        such as:

            29 Aug 2026, 11:03:24 pm

        If Date can parse it, format it.
        Otherwise preserve the original value.
    */

    const date =
        new Date(timestamp);


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {

        return String(timestamp);

    }


    return new Intl.DateTimeFormat(
        undefined,
        {
            day: "2-digit",
            month: "short",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit",
            second: "2-digit",
            hour12: true
        }
    ).format(date);

}


/* ============================================================
   RENDER HISTORY TABLE
   ============================================================ */

function renderHistoryRows(decisions) {

    /*
        IMPORTANT:

        The HTML table uses:

            <tbody id="history-table-body">

        Do NOT use:

            decision-history-body
    */


    const tableBody =
        document.getElementById(
            "history-table-body"
        );


    const emptyState =
        document.getElementById(
            "history-empty"
        );


    const tableContainer =
        document.getElementById(
            "history-table-container"
        );


    if (!tableBody) {

        console.error(
            "Decision history table body not found."
        );

        return;

    }


    tableBody.innerHTML =
        "";


    if (
        !Array.isArray(decisions) ||
        decisions.length === 0
    ) {

        if (emptyState) {

            emptyState.hidden =
                false;

        }


        if (tableContainer) {

            tableContainer.hidden =
                true;

        }


        return;

    }


    if (emptyState) {

        emptyState.hidden =
            true;

    }


    if (tableContainer) {

        tableContainer.hidden =
            false;

    }


    decisions.forEach(
        decision => {

            const row =
                document.createElement(
                    "tr"
                );


            const amount =
                Number(
                    decision.amount || 0
                );


            const riskPercentage =
                Number(
                    decision.risk_percentage ??
                    (
                        Number(decision.risk_score || 0) *
                        100
                    )
                );


            const signals =
                Number(
                    decision.signals ??
                    (
                        Array.isArray(
                            decision.reasons
                        )
                            ? decision.reasons.length
                            : 0
                    )
                );


            const decisionClass =
                String(
                    decision.decision || ""
                ).toLowerCase();


            const riskClass =
                String(
                    decision.risk_level || ""
                ).toLowerCase();


            const formattedTimestamp =
                formatHistoryTimestamp(
                    decision.timestamp
                );


            row.innerHTML = `

                <td>
                    ${escapeHistoryText(
                        formattedTimestamp
                    )}
                </td>

                <td>
                    ₹${amount.toLocaleString(
                        "en-IN",
                        {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                        }
                    )}
                </td>

                <td>
                    ${Number.isFinite(riskPercentage)
                        ? riskPercentage.toFixed(2)
                        : "0.00"}%
                </td>

                <td>
                    <span class="risk-badge ${escapeHistoryText(riskClass)}">
                        ${escapeHistoryText(
                            decision.risk_level ||
                            "UNKNOWN"
                        )}
                    </span>
                </td>

                <td>
                    <span class="decision-badge ${escapeHistoryText(decisionClass)}">
                        ${escapeHistoryText(
                            decision.decision ||
                            "UNKNOWN"
                        )}
                    </span>
                </td>

                <td>
                    ${signals}
                    ${
                        signals === 1
                            ? "signal"
                            : "signals"
                    }
                </td>

            `;


            tableBody.appendChild(
                row
            );

        }
    );

}


/* ============================================================
   HISTORY ERROR
   ============================================================ */

function showHistoryError() {

    const tableBody =
        document.getElementById(
            "history-table-body"
        );


    const emptyState =
        document.getElementById(
            "history-empty"
        );


    const tableContainer =
        document.getElementById(
            "history-table-container"
        );


    if (tableBody) {

        tableBody.innerHTML = `

            <tr>

                <td
                    colspan="6"
                    class="empty-history"
                >

                    Unable to load decision history.

                </td>

            </tr>

        `;

    }


    if (emptyState) {

        emptyState.hidden =
            true;

    }


    if (tableContainer) {

        tableContainer.hidden =
            false;

    }

}


/* ============================================================
   SAFE HISTORY TEXT
   ============================================================ */

function escapeHistoryText(value) {

    const element =
        document.createElement(
            "div"
        );


    element.textContent =
        String(value);


    return element.innerHTML;

}


/* ============================================================
   REFRESH HISTORY BUTTON
   ============================================================ */

function initializeHistoryButton() {

    const button =
        getElement(
            "refresh-history"
        );


    if (!button) {

        console.warn(
            "Refresh history button not found."
        );

        return;

    }


    /*
        Prevent duplicate event listeners.
    */

    if (
        button.dataset.historyRefreshInitialized ===
        "true"
    ) {

        return;

    }


    button.dataset.historyRefreshInitialized =
        "true";


    button.addEventListener(
        "click",
        async function () {

            /*
                Prevent multiple simultaneous refreshes.
            */

            if (
                button.disabled
            ) {

                return;

            }


            const originalText =
                button.textContent;


            button.disabled =
                true;


            button.textContent =
                "Refreshing...";


            button.setAttribute(
                "aria-busy",
                "true"
            );


            try {

                console.log(
                    "Refreshing Decision History..."
                );


                await loadDecisionHistory();


                console.log(
                    "Decision History refreshed successfully."
                );

            }

            catch (error) {

                console.error(
                    "Decision History refresh failed:",
                    error
                );

            }


            finally {

                button.disabled =
                    false;


                button.textContent =
                    originalText;


                button.removeAttribute(
                    "aria-busy"
                );

            }

        }
    );


    console.log(
        "Decision History refresh button initialized."
    );

}


/* ============================================================
   NAVIGATION
   ============================================================ */

function initializeNavigation() {

    const navigationItems =
        document.querySelectorAll(
            ".nav-item"
        );


    const sections = {

        overview:
            document.getElementById(
                "overview-section"
            ),

        analysis:
            document.getElementById(
                "analysis-section"
            ),

        signals:
            document.getElementById(
                "signals-section"
            ),

        history:
            document.getElementById(
                "history-section"
            )

    };


    const pageTitle =
        document.getElementById(
            "page-title"
        );


    console.log(
        "Navigation initialized."
    );


    console.log(
        "Sections:",
        sections
    );


    navigationItems.forEach(
        item => {

            item.addEventListener(
                "click",
                async function () {

                    const sectionName =
                        item.dataset.section;


                    console.log(
                        "NAV CLICK:",
                        sectionName
                    );


                    /* ----------------------------------------
                       REMOVE ACTIVE FROM ALL NAV ITEMS
                    ---------------------------------------- */

                    navigationItems.forEach(
                        nav => {

                            nav.classList.remove(
                                "active"
                            );

                        }
                    );


                    /* ----------------------------------------
                       ADD ACTIVE TO CURRENT ITEM
                    ---------------------------------------- */

                    item.classList.add(
                        "active"
                    );


                    /* ----------------------------------------
                       HIDE ALL SECTIONS
                    ---------------------------------------- */

                    Object.values(
                        sections
                    ).forEach(
                        section => {

                            if (section) {

                                section.classList.add(
                                    "hidden-section"
                                );

                            }

                        }
                    );


                    /* ----------------------------------------
                       SHOW SELECTED SECTION
                    ---------------------------------------- */

                    const selected =
                        sections[
                            sectionName
                        ];


                    if (!selected) {

                        console.error(
                            "SECTION NOT FOUND:",
                            sectionName
                        );

                        return;
                    }


                    selected.classList.remove(
                        "hidden-section"
                    );


                    console.log(
                        "SECTION SHOWN:",
                        sectionName
                    );


                    /* ----------------------------------------
                       UPDATE PAGE TITLE
                    ---------------------------------------- */

                    const titles = {

                        overview:
                            "Risk Overview",

                        analysis:
                            "Transaction Analysis",

                        signals:
                            "Risk Signals",

                        history:
                            "Decision History"

                    };


                    if (pageTitle) {

                        pageTitle.textContent =
                            titles[
                                sectionName
                            ] ||
                            "Risk Overview";

                    }


                    /* ----------------------------------------
                       LOAD DECISION HISTORY
                    ---------------------------------------- */

                    if (
                        sectionName ===
                        "history"
                    ) {

                        console.log(
                            "Loading decision history..."
                        );


                        await loadDecisionHistory();

                    }


                    /* ----------------------------------------
                       SCROLL TOP
                    ---------------------------------------- */

                    window.scrollTo({
                        top: 0,
                        behavior: "smooth"
                    });

                }
            );

        }
    );
}


/* ============================================================
   NAVIGATION INITIAL STATE
   ============================================================ */

function initializeSections() {

    const sections = [

        getElement(
            "overview-section"
        ),

        getElement(
            "analysis-section"
        ),

        getElement(
            "signals-section"
        ),

        getElement(
            "history-section"
        )

    ];


    sections.forEach(
        section => {

            if (section) {

                section.classList.add(
                    "hidden-section"
                );

            }

        }
    );


    const overview =
        getElement(
            "overview-section"
        );


    if (overview) {

        overview.classList.remove(
            "hidden-section"
        );

    }


    const navigationItems =
        document.querySelectorAll(
            ".nav-item"
        );


    navigationItems.forEach(
        item => {

            item.classList.remove(
                "active"
            );


            if (
                item.dataset.section ===
                "overview"
            ) {

                item.classList.add(
                    "active"
                );

            }

        }
    );


    const pageTitle =
        getElement(
            "page-title"
        );


    if (pageTitle) {

        pageTitle.textContent =
            "Risk Overview";

    }

}


/* ============================================================
   AMOUNT HISTORY INPUT HANDLING
   ============================================================ */

function initializeAmountHistoryInput() {

    const input =
        getElement(
            "amount-history"
        );


    if (!input) {
        return;
    }


    /*
        Keep arbitrary decimal values.

        Example:

            2.6765404593780824

        is allowed.

        Do not force:

            2.6
            2.7
    */

    input.setAttribute(
        "step",
        "any"
    );


    input.addEventListener(
        "input",
        function () {

            const value =
                Number(
                    input.value
                );


            if (
                input.value !== "" &&
                (
                    !Number.isFinite(value) ||
                    value < 0
                )
            ) {

                input.setCustomValidity(
                    "Enter a valid non-negative number."
                );

            }

            else {

                input.setCustomValidity(
                    ""
                );

            }

        }
    );

}


/* ============================================================
   KEYBOARD SHORTCUT
   ============================================================ */

function initializeKeyboardShortcuts() {

    document.addEventListener(
        "keydown",
        function (event) {

            /*
                Ctrl + Enter analyzes transaction.
            */

            if (
                event.ctrlKey &&
                event.key === "Enter"
            ) {

                const form =
                    getElement(
                        "risk-form"
                    );


                if (form) {

                    form.requestSubmit();

                }

            }

        }
    );

}


/* ============================================================
   INITIALIZATION
   ============================================================ */

async function initializeRazorShield() {

    console.log(
        "Initializing RazorShield AI..."
    );


    /*
        ========================================================
        STEP 1
        INITIALIZE NAVIGATION FIRST
        ========================================================
    */

    try {

        initializeNavigation();


        console.log(
            "✓ Navigation initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ Navigation initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 2
        INITIALIZE PAGE VISIBILITY
        ========================================================
    */

    try {

        initializeSections();


        console.log(
            "✓ Sections initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ Section initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 3
        INITIALIZE DEMO BUTTONS
        ========================================================
    */

    try {

        initializeDemoButtons();


        console.log(
            "✓ Demo buttons initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ Demo button initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 4
        INITIALIZE RISK FORM
        ========================================================
    */

    try {

        initializeRiskForm();


        console.log(
            "✓ Risk form initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ Risk form initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 5
        INITIALIZE HISTORY BUTTON
        ========================================================
    */

    try {

        initializeHistoryButton();


        console.log(
            "✓ History button initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ History button initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 6
        OTHER UI
        ========================================================
    */

    try {

        initializeAmountHistoryInput();


        console.log(
            "✓ Amount history initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ Amount history initialization failed:",
            error
        );

    }


    try {

        initializeKeyboardShortcuts();


        console.log(
            "✓ Keyboard shortcuts initialized"
        );

    }

    catch (error) {

        console.error(
            "✗ Keyboard shortcuts initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 7
        API HEALTH
        ========================================================
    */

    try {

        await checkApiHealth();

    }

    catch (error) {

        console.error(
            "API health initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 8
        LOAD DEMO TRANSACTIONS
        ========================================================
    */

    try {

        await loadDemoTransactions();

    }

    catch (error) {

        console.error(
            "Demo transactions initialization failed:",
            error
        );

    }


    /*
        ========================================================
        STEP 9
        LOAD COMBINED DECISION HISTORY
        ========================================================
    */

    try {

        await loadDecisionHistory();

    }

    catch (error) {

        console.error(
            "Decision history initialization failed:",
            error
        );

    }


    console.log(
        "RazorShield AI initialized successfully."
    );

}


/* ============================================================
   START APPLICATION
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        initializeRazorShield();

    }
);