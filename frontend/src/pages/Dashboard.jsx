import React, { useEffect, useMemo, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend);

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000/api";

const pageStyle = {
  minHeight: "100vh",
  background: "#f3f6fb",
  padding: "24px",
  fontFamily: "Inter, Arial, sans-serif",
};

const cardStyle = {
  background: "#fff",
  borderRadius: "12px",
  padding: "16px",
  boxShadow: "0 4px 18px rgba(0, 0, 0, 0.08)",
};

function Dashboard() {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchViolations = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await fetch(`${API_BASE_URL}/violations`);
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }
        const data = await response.json();
        setViolations(Array.isArray(data) ? data : []);
      } catch (err) {
        setError(err.message || "Failed to fetch violations.");
      } finally {
        setLoading(false);
      }
    };

    fetchViolations();
  }, []);

  const chartData = useMemo(() => {
    const byType = {};
    for (const item of violations) {
      const key = item.violationType || "Unknown";
      byType[key] = (byType[key] || 0) + 1;
    }

    const labels = Object.keys(byType);
    const counts = Object.values(byType);

    return {
      labels,
      counts,
    };
  }, [violations]);

  const barData = {
    labels: chartData.labels,
    datasets: [
      {
        label: "Violations",
        data: chartData.counts,
        backgroundColor: "#3b82f6",
        borderRadius: 8,
      },
    ],
  };

  const doughnutData = {
    labels: chartData.labels,
    datasets: [
      {
        data: chartData.counts,
        backgroundColor: ["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6"],
      },
    ],
  };

  return (
    <div style={pageStyle}>
      <div style={{ marginBottom: "20px" }}>
        <h1 style={{ margin: 0, fontSize: "28px", color: "#0f172a" }}>Traffic Violations Dashboard</h1>
        <p style={{ margin: "8px 0 0", color: "#475569" }}>
          Real-time summary of detected traffic violations
        </p>
      </div>

      {loading && <p>Loading violations...</p>}
      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}

      {!loading && !error && (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: "16px",
              marginBottom: "20px",
            }}
          >
            <div style={cardStyle}>
              <h3 style={{ marginTop: 0 }}>Violations by Type</h3>
              <Bar data={barData} options={{ responsive: true, plugins: { legend: { display: false } } }} />
            </div>

            <div style={cardStyle}>
              <h3 style={{ marginTop: 0 }}>Distribution</h3>
              <Doughnut data={doughnutData} />
            </div>
          </div>

          <div style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Violation Records</h3>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ background: "#eef2ff", textAlign: "left" }}>
                    <th style={{ padding: "10px" }}>Vehicle Number</th>
                    <th style={{ padding: "10px" }}>Violation Type</th>
                    <th style={{ padding: "10px" }}>Time</th>
                    <th style={{ padding: "10px" }}>Image</th>
                  </tr>
                </thead>
                <tbody>
                  {violations.length === 0 ? (
                    <tr>
                      <td colSpan="4" style={{ padding: "12px", textAlign: "center", color: "#64748b" }}>
                        No violations found.
                      </td>
                    </tr>
                  ) : (
                    violations.map((item) => (
                      <tr key={item._id || `${item.vehicleNumber}-${item.timestamp}`} style={{ borderBottom: "1px solid #e2e8f0" }}>
                        <td style={{ padding: "10px" }}>{item.vehicleNumber || "N/A"}</td>
                        <td style={{ padding: "10px", textTransform: "capitalize" }}>{item.violationType || "N/A"}</td>
                        <td style={{ padding: "10px" }}>
                          {item.timestamp ? new Date(item.timestamp).toLocaleString() : "N/A"}
                        </td>
                        <td style={{ padding: "10px" }}>
                          {item.imageUrl ? (
                            <a href={item.imageUrl} target="_blank" rel="noreferrer">
                              View
                            </a>
                          ) : (
                            "N/A"
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;

