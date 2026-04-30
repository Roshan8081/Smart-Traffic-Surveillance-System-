const Violation = require("../models/Violation");

function normalizeViolationPayload(body = {}) {
  const vehicleNumber =
    body.vehicleNumber ||
    body.plateNumber ||
    (body.meta && body.meta.plate_text) ||
    "N/A";

  const violationType = body.violationType || body.type || "unknown";

  let imageUrl = body.imageUrl || body.imagePath || "";
  const imgStr = String(imageUrl || "");
  const marker = /detection[\\/]+violations[\\/]+(.+)$/i.exec(imgStr);
  if (marker && marker[1]) {
    const rel = marker[1].replace(/\\/g, "/");
    imageUrl = `http://localhost:5000/evidence/${rel}`;
  }

  const timestamp = body.timestamp ? new Date(body.timestamp) : new Date();

  return {
    vehicleNumber: String(vehicleNumber).trim() || "N/A",
    violationType: String(violationType).trim() || "unknown",
    timestamp,
    imageUrl: String(imageUrl || "").trim(),
  };
}

async function createViolation(req, res, next) {
  try {
    const data = normalizeViolationPayload(req.body);
    if (!data.violationType) {
      return res.status(400).json({ message: "violationType is required." });
    }
    const violation = await Violation.create(data);
    return res.status(201).json(violation);
  } catch (error) {
    return next(error);
  }
}

async function getViolations(_req, res, next) {
  try {
    const violations = await Violation.find().sort({ timestamp: -1 });
    return res.status(200).json(violations);
  } catch (error) {
    return next(error);
  }
}

module.exports = {
  createViolation,
  getViolations,
};

