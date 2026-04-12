import { NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await callPythonBridge("credit_demand_by_grade", {});
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "credit demand by grade failed" },
      { status: 500 }
    );
  }
}
