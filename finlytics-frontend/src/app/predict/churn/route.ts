import { NextRequest, NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const data = await callPythonBridge("predict_churn", payload);
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "churn prediction failed" },
      { status: 500 }
    );
  }
}
