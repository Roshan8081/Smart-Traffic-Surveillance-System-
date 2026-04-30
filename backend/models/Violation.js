const mongoose = require("mongoose");

const violationSchema = new mongoose.Schema(
  {
    vehicleNumber: {
      type: String,
      required: true,
      trim: true,
    },
    violationType: {
      type: String,
      required: true,
      trim: true,
    },
    timestamp: {
      type: Date,
      required: true,
      default: Date.now,
    },
    imageUrl: {
      type: String,
      default: "",
      trim: true,
    },
  },
  {
    versionKey: false,
  }
);

module.exports = mongoose.model("Violation", violationSchema);

