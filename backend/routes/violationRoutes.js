const express = require("express");
const {
  createViolation,
  getViolations,
} = require("../controllers/violationController");

const router = express.Router();

// POST /api/violations
router.post("/", createViolation);

// GET /api/violations
router.get("/", getViolations);

module.exports = router;

