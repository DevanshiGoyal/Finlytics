import { NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await callPythonBridge("deposit_leaderboard", {});
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "deposit leaderboard failed" },
      { status: 500 }
    );
  }
}
