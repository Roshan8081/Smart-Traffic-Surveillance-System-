const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const path = require("path");

const violationRoutes = require("./routes/violationRoutes");
const errorHandler = require("./middleware/errorHandler");

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(
  "/evidence",
  express.static(path.resolve(__dirname, "..", "detection", "violations"))
);

// Routes
app.use("/api/violations", violationRoutes);

// Health check
app.get("/api/health", (_req, res) => {
  res.status(200).json({ ok: true, service: "smart-traffic-backend" });
});

app.use(errorHandler);

const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/smart_traffic_system";

async function startServer() {
  try {
    await mongoose.connect(MONGO_URI);
    console.log("[DB] MongoDB connected");

    app.listen(PORT, () => {
      console.log(`[SERVER] Running on http://localhost:${PORT}`);
    });
  } catch (error) {
    console.error("[FATAL] Failed to start server:", error.message);
    process.exit(1);
  }
}

startServer();

