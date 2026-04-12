import { NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await callPythonBridge("anomaly_timeseries", {});
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "anomaly timeseries failed" },
      { status: 500 }
    );
  }
}
